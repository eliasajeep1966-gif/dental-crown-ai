#!/usr/bin/env python3
"""Prepare one local exocad case as a MADCrowner-format training sample.

This is a candidate data-preparation tool, not a clinical or training approval.
It intentionally supports the currently validated FDI 46 XML schema only.

The NumPy DPSR implementation follows the equations and rasterization conventions
in third_party/MADCrowner/models/SAP/src/dpsr.py and utils.py (SAP MIT license;
see third_party/MADCrowner/models/SAP/LICENSE). Keep the upstream notice with this
repository when redistributing this file.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from inspect_case_alignment import (  # noqa: E402
    load_geometry,
    resolve_folder,
    resolve_rar,
    select_jaw_scan,
    surface_distance_report,
)


class PreparationError(RuntimeError):
    """A validation failure that should stop sample creation."""


def _check_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise PreparationError(f"{name} يحتوي على قيم غير منتهية.")


def weld_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Create connected topology from STL's duplicated per-triangle vertices."""
    result = mesh.copy()
    result.merge_vertices(digits_vertex=8)
    result.remove_unreferenced_vertices()
    if len(result.vertices) == 0 or len(result.faces) == 0:
        raise PreparationError("الشبكة فارغة بعد دمج الرؤوس.")
    result.fix_normals()
    return result


def point_to_polyline_distances(points: np.ndarray, ordered_line: np.ndarray,
                                chunk_size: int = 512) -> np.ndarray:
    """Exact distances from points to a closed, ordered 3-D polyline."""
    points = np.asarray(points, dtype=np.float64)
    line = np.asarray(ordered_line, dtype=np.float64)
    if len(line) < 3:
        raise PreparationError("خط Margin يحتاج ثلاث نقاط على الأقل.")
    starts = line
    vectors = np.roll(line, -1, axis=0) - line
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    valid = lengths_squared > 1e-20
    starts, vectors, lengths_squared = starts[valid], vectors[valid], lengths_squared[valid]
    if len(starts) < 3:
        raise PreparationError("خط Margin يحتوي مقاطع غير صالحة.")

    output = np.empty(len(points), dtype=np.float64)
    for begin in range(0, len(points), chunk_size):
        current = points[begin:begin + chunk_size]
        delta = current[:, None, :] - starts[None, :, :]
        t = np.einsum("psd,sd->ps", delta, vectors) / lengths_squared[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = starts[None, :, :] + t[:, :, None] * vectors[None, :, :]
        dist2 = np.einsum("psd,psd->ps", current[:, None, :] - closest,
                          current[:, None, :] - closest)
        output[begin:begin + len(current)] = np.sqrt(np.min(dist2, axis=1))
    return output


def label_mesh_vertices(mesh_vertices: np.ndarray, margin_points: np.ndarray,
                        units_per_mm: float, direct_tolerance_mm: float,
                        max_snap_mm: float) -> tuple[np.ndarray, dict]:
    """Mark vertices near the margin polyline and snap each XML point to a vertex."""
    distances_units = point_to_polyline_distances(mesh_vertices, margin_points)
    direct = distances_units <= direct_tolerance_mm * units_per_mm
    nearest_dist_units, nearest_indices = cKDTree(mesh_vertices).query(margin_points, k=1)
    nearest_dist_mm = nearest_dist_units / units_per_mm
    if float(np.max(nearest_dist_mm)) > max_snap_mm:
        raise PreparationError(
            f"بعض نقاط Margin أبعد من حد إسقاط الرؤوس المسموح ({max_snap_mm:g} mm)."
        )
    labels = direct.astype(np.uint8)
    labels[np.asarray(nearest_indices, dtype=np.int64)] = 1
    return labels, {
        "direct_tolerance_mm": float(direct_tolerance_mm),
        "directly_near_polyline_vertex_count": int(np.count_nonzero(direct)),
        "snapped_margin_sample_count": int(len(margin_points)),
        "unique_snapped_vertex_count": int(np.unique(nearest_indices).size),
        "max_margin_sample_to_nearest_vertex_mm": float(np.max(nearest_dist_mm)),
        "positive_vertex_count": int(np.count_nonzero(labels)),
        "positive_vertex_fraction": float(np.mean(labels)),
        "method": "distance-to-closed-ordered-polyline plus nearest-vertex snap",
    }


def mean_curvature_vtk(mesh: trimesh.Trimesh) -> np.ndarray:
    """Use the VTK mean-curvature operator, as in the upstream SAP helper."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise PreparationError("حزمة vtk مطلوبة لحساب curvatures.") from exc

    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for vertex in np.asarray(mesh.vertices, dtype=np.float64):
        points.InsertNextPoint(float(vertex[0]), float(vertex[1]), float(vertex[2]))

    triangles = vtk.vtkCellArray()
    for face in np.asarray(mesh.faces, dtype=np.int64):
        triangle = vtk.vtkTriangle()
        ids = triangle.GetPointIds()
        ids.SetId(0, int(face[0]))
        ids.SetId(1, int(face[1]))
        ids.SetId(2, int(face[2]))
        triangles.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    curvature_filter = vtk.vtkCurvatures()
    curvature_filter.SetInputData(polydata)
    curvature_filter.SetCurvatureTypeToMean()
    curvature_filter.Update()
    vtk_values = curvature_filter.GetOutput().GetPointData().GetScalars()
    if vtk_values is None or vtk_values.GetNumberOfTuples() != len(mesh.vertices):
        raise PreparationError("لم ينتج VTK قيمة curvature لكل رأس في crown.ply.")
    curvatures = np.asarray(vtk_to_numpy(vtk_values), dtype=np.float32).reshape(-1)
    _check_finite("curvatures", curvatures)
    return curvatures


def _trilinear_grid_sample(grid: np.ndarray, points01: np.ndarray) -> np.ndarray:
    """Trilinearly sample a cubic periodic grid using SAP's [0,1] coordinates."""
    res = grid.shape[0]
    points = np.clip(np.asarray(points01, dtype=np.float64), 0.0, 1.0 - 1e-9)
    scaled = points * res
    lower = np.floor(scaled).astype(np.int64)
    upper = (np.ceil(scaled).astype(np.int64)) % res
    frac = scaled - lower
    values = np.zeros(len(points), dtype=np.float64)
    for bits in itertools.product((0, 1), repeat=3):
        index = np.where(np.asarray(bits, dtype=bool), upper, lower)
        weight_axis = np.where(np.asarray(bits, dtype=bool), frac, 1.0 - frac)
        weights = np.prod(weight_axis, axis=1)
        values += grid[index[:, 0], index[:, 1], index[:, 2]] * weights
    return values


def dpsr_numpy(points01: np.ndarray, normals: np.ndarray, resolution: int = 128) -> tuple[np.ndarray, str]:
    """CPU NumPy DPSR field (sig=0, shift=True, scale=True), SAP-compatible."""
    points = np.asarray(points01, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise PreparationError("نقاط PSR وnormals يجب أن تكونا بالشكل نفسه (N,3).")
    if len(points) < 4:
        raise PreparationError("عدد رؤوس crown غير كافٍ لحساب PSR.")
    _check_finite("PSR points", points)
    _check_finite("PSR normals", normals)
    if np.any(points < -1e-7) or np.any(points > 1.0 + 1e-7):
        raise PreparationError("إحداثيات PSR خرجت عن المجال [0,1].")
    points = np.clip(points, 0.0, 1.0 - 1e-9)
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths < 1e-10):
        raise PreparationError("يوجد normal صفري لا يمكن استخدامه في PSR.")
    normals = normals / normal_lengths[:, None]

    # Trilinear point rasterization; periodic wrap matches SAP point_rasterize.
    scaled = points * resolution
    lower = np.floor(scaled).astype(np.int64)
    upper = (np.ceil(scaled).astype(np.int64)) % resolution
    frac = scaled - lower
    raster = np.zeros((resolution, resolution, resolution, 3), dtype=np.float64)
    for bits in itertools.product((0, 1), repeat=3):
        bit_array = np.asarray(bits, dtype=bool)
        index = np.where(bit_array, upper, lower)
        weights = np.prod(np.where(bit_array, frac, 1.0 - frac), axis=1)
        np.add.at(raster, (index[:, 0], index[:, 1], index[:, 2]),
                  normals * weights[:, None])

    spectrum = np.fft.rfftn(raster, axes=(0, 1, 2))
    freq_x = np.fft.fftfreq(resolution, d=1.0 / resolution)
    freq_y = np.fft.fftfreq(resolution, d=1.0 / resolution)
    freq_z = np.fft.rfftfreq(resolution, d=1.0 / resolution)
    wx, wy, wz = np.meshgrid(freq_x, freq_y, freq_z, indexing="ij")
    omega = 2.0 * np.pi * np.stack((wx, wy, wz), axis=-1)
    divergence = -1j * np.sum(spectrum * omega, axis=-1)
    laplacian = -np.sum(omega * omega, axis=-1)
    phi_spectrum = divergence / (laplacian + 1e-6)
    phi_spectrum[0, 0, 0] = 0.0
    phi = np.fft.irfftn(phi_spectrum, s=(resolution,) * 3, axes=(0, 1, 2)).real

    # Match DPSR's shift and scale: mean field at input points is zero and the
    # value at the grid origin is normalized to -0.5.
    mean_at_points = float(np.mean(_trilinear_grid_sample(phi, points)))
    phi -= mean_at_points
    origin_value = float(phi[0, 0, 0])
    scale_mode = "SAP_origin_value"
    if abs(origin_value) < 1e-12:
        # Symmetric shapes can have exactly zero field at the grid origin. SAP's
        # original division would become undefined; use a clearly reported robust
        # fallback instead of writing NaN/Inf values.
        scale_value = float(np.quantile(np.abs(phi), 0.99))
        scale_mode = "p99_abs_fallback_origin_was_zero"
    else:
        scale_value = abs(origin_value)
    if scale_value < 1e-12:
        raise PreparationError("حقل PSR كله قريب من الصفر ولا يمكن تطبيعه.")
    phi = -phi / scale_value * 0.5
    _check_finite("PSR field", phi)
    return phi.astype(np.float16), scale_mode


def crop_mesh_around(mesh: trimesh.Trimesh, center: np.ndarray,
                     crop_size_units: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Crop triangles wholly inside the same cubic window used by the loader."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    inside = np.all(np.abs(vertices - center[None, :]) <= crop_size_units / 2.0, axis=1)
    face_mask = np.all(inside[faces], axis=1)
    kept_faces_old = faces[face_mask]
    if len(kept_faces_old) == 0:
        raise PreparationError("قصّ mandibular حول التاج لم ينتج وجوهًا؛ افحص إطار المسح وحجم القص.")
    used_old_indices = np.unique(kept_faces_old.reshape(-1))
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[used_old_indices] = np.arange(len(used_old_indices), dtype=np.int64)
    cropped = trimesh.Trimesh(
        vertices=vertices[used_old_indices],
        faces=old_to_new[kept_faces_old],
        process=False,
    )
    if len(cropped.vertices) != len(used_old_indices) or len(cropped.faces) == 0:
        raise PreparationError("قصّ PNA غير صالح.")
    return cropped, used_old_indices


def _write_ply(mesh: trimesh.Trimesh, path: Path) -> None:
    mesh.export(str(path), file_type="ply")
    reopened = trimesh.load(str(path), process=False)
    if isinstance(reopened, trimesh.Scene):
        reopened = reopened.dump(concatenate=True)
    if not isinstance(reopened, trimesh.Trimesh):
        raise PreparationError("تعذّر إعادة قراءة ملف PLY الناتج.")
    if len(reopened.vertices) != len(mesh.vertices) or len(reopened.faces) != len(mesh.faces):
        raise PreparationError("تغيّر عدد الرؤوس أو الوجوه بعد حفظ PLY.")
    if not np.allclose(reopened.vertices, mesh.vertices, atol=2e-5, rtol=0.0):
        raise PreparationError("تغيّر ترتيب/إحداثيات الرؤوس بعد حفظ PLY؛ أُوقف ربط الوسوم.")


def _resolve_case(case_path: Path, work_dir: Path, constructioninfo: str | None,
                  crown_name: str | None, modeldie_name: str | None):
    if case_path.is_dir():
        return resolve_folder(case_path, work_dir, constructioninfo, crown_name, modeldie_name)
    if case_path.is_file() and case_path.suffix.casefold() == ".rar":
        return resolve_rar(case_path, work_dir, constructioninfo, crown_name, modeldie_name)
    raise PreparationError("--case يجب أن يكون مجلد حالة أو أرشيف RAR محليًا.")


def build_sample(args: argparse.Namespace) -> dict:
    case_path = Path(args.case).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    if output_path == REPO_ROOT or REPO_ROOT in output_path.parents:
        raise PreparationError("احفظ بيانات الحالة خارج مجلد المشروع حتى لا تُرفع إلى GitHub.")
    if output_path.exists():
        raise PreparationError("مجلد الإخراج موجود مسبقًا؛ اختر اسمًا جديدًا لحماية النتائج القديمة.")
    if case_path.is_dir() and (output_path == case_path or case_path in output_path.parents):
        raise PreparationError("مجلد الإخراج يجب أن يكون خارج مجلد الحالة.")
    if args.units_per_mm <= 0 or args.crop_size_mm <= 0 or args.margin_tolerance_mm <= 0:
        raise PreparationError("قيم الوحدات وحجم القص وسماحية الوسم يجب أن تكون موجبة.")
    if args.psr_resolution < 32 or args.psr_resolution & (args.psr_resolution - 1):
        raise PreparationError("دقة PSR يجب أن تكون قوة للعدد 2 وألا تقل عن 32.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".{output_path.name}.building-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix="dental_case_local_") as temporary:
            work_dir = Path(temporary)
            xml_path, parsed, scans, crown_path, _die_path = _resolve_case(
                case_path, work_dir, args.constructioninfo, args.crown, args.modeldie
            )
            margin = np.asarray(parsed["margin_points"], dtype=np.float64)
            _check_finite("Margin", margin)
            _, jaw_item = select_jaw_scan(scans, args.jaw_hint)
            if jaw_item["record"]["matrix"] is None:
                raise PreparationError("مسح الفك لا يحتوي TransformationMatrix.")

            crown_loaded = load_geometry(crown_path)
            if not isinstance(crown_loaded, trimesh.Trimesh) or len(crown_loaded.faces) == 0:
                raise PreparationError("crown STL لا يحتوي شبكة مثلثية صالحة.")
            crown = weld_mesh(crown_loaded)
            if not crown.is_watertight:
                raise PreparationError("crown STL ليس watertight بعد دمج الرؤوس؛ أوقفنا PSR لتجنب هدف حجمي مضلل.")
            crown.fix_normals()
            if crown.volume < 0:
                crown.invert()
            unique_edges = np.asarray(crown.edges_unique)
            edge_inverse = np.asarray(crown.edges_unique_inverse)
            edge_use_counts = np.bincount(edge_inverse, minlength=len(unique_edges))
            crown_boundary_edge_count = int(np.count_nonzero(edge_use_counts == 1))

            jaw_loaded = load_geometry(jaw_item["path"])
            if not isinstance(jaw_loaded, trimesh.Trimesh) or len(jaw_loaded.faces) == 0:
                raise PreparationError("مسح الفك لا يحتوي شبكة مثلثية صالحة.")
            jaw = weld_mesh(jaw_loaded)
            # The current local review selected ScanFile transpose_forward for the
            # jaw; crown and XML Margin stay raw. This is explicit in the manifest.
            jaw.apply_transform(np.asarray(jaw_item["record"]["matrix"], dtype=np.float64).T)

            crown_distance = surface_distance_report(crown, margin, args.units_per_mm)
            jaw_distance = surface_distance_report(jaw, margin, args.units_per_mm)
            if crown_distance["max_mm"] > args.max_crown_margin_mm:
                raise PreparationError("Margin لا يطابق سطح crown ضمن الحد المطلوب؛ لم تُنشأ عينة.")
            if jaw_distance["max_mm"] > args.max_jaw_margin_mm:
                raise PreparationError("Margin بعيد عن سطح الفك بعد التحويل المختار؛ لم تُنشأ عينة.")

            crown_center = (np.asarray(crown.vertices).min(axis=0) +
                            np.asarray(crown.vertices).max(axis=0)) / 2.0
            crop_size_units = args.crop_size_mm * args.units_per_mm
            normalized_crown = (np.asarray(crown.vertices, dtype=np.float64) - crown_center) / (crop_size_units / 2.0)
            if np.max(np.abs(normalized_crown)) > 1.02:
                raise PreparationError("التاج لا يلائم حجم القص 20 mm؛ افحص الوحدات أو --crop-size-mm.")
            psr_points = (normalized_crown + 1.0) / 2.0
            crown_normals = np.asarray(crown.vertex_normals, dtype=np.float64)
            if len(crown_normals) != len(crown.vertices):
                raise PreparationError("تعذّر حساب normal لكل رأس تاج.")

            pna_crop, pna_original_indices = crop_mesh_around(
                jaw, crown_center, crop_size_units
            )
            boundary_labels, pna_label_report = label_mesh_vertices(
                np.asarray(pna_crop.vertices), margin, args.units_per_mm,
                args.margin_tolerance_mm, args.max_snap_mm,
            )
            if not np.any(boundary_labels):
                raise PreparationError("لم تُسند أي تسمية Margin إلى رؤوس PNA.")
            crown_margin_labels, crown_label_report = label_mesh_vertices(
                np.asarray(crown.vertices), margin, args.units_per_mm,
                args.margin_tolerance_mm, args.max_crown_snap_mm,
            )
            curvatures = mean_curvature_vtk(crown)
            psr, psr_scale_mode = dpsr_numpy(
                psr_points, crown_normals, args.psr_resolution
            )

            _write_ply(crown, staging / "crown.ply")
            _write_ply(pna_crop, staging / "pna_crop.ply")
            np.save(staging / "boundary_labels.npy", boundary_labels.astype(np.uint8))
            with h5py.File(staging / "crown_attributes.h5", "w") as h5:
                h5.create_dataset("curvatures", data=curvatures.astype(np.float32), compression="gzip")
                h5.create_dataset("whetherismarginline", data=crown_margin_labels.astype(np.uint8), compression="gzip")
                h5.attrs["curvature_method"] = "VTK vtkCurvatures mean"
                h5.attrs["margin_label_method"] = "distance to XML Margin polyline plus nearest-vertex snap"
            np.savez_compressed(staging / "psr_my.npz", psr=psr)

            # Verify on-disk lengths and keys against the actual MADCrowner loader.
            saved_crown = trimesh.load(str(staging / "crown.ply"), process=False)
            saved_pna = trimesh.load(str(staging / "pna_crop.ply"), process=False)
            saved_labels = np.load(staging / "boundary_labels.npy", allow_pickle=False)
            if len(saved_crown.vertices) != len(curvatures) or len(saved_crown.vertices) != len(crown_margin_labels):
                raise PreparationError("طول صفات crown لا يطابق رؤوس PLY الناتج.")
            if len(saved_pna.vertices) != len(saved_labels):
                raise PreparationError("طول boundary_labels لا يطابق رؤوس PNA PLY الناتج.")
            with h5py.File(staging / "crown_attributes.h5", "r") as h5:
                if set(h5.keys()) != {"curvatures", "whetherismarginline"}:
                    raise PreparationError("مفاتيح HDF5 لا تطابق ما يقرأه MADCrowner.")
                if h5["curvatures"].shape != (len(saved_crown.vertices),):
                    raise PreparationError("شكل curvatures غير متوافق مع crown vertices.")
                if h5["whetherismarginline"].shape != (len(saved_crown.vertices),):
                    raise PreparationError("شكل whetherismarginline غير متوافق مع crown vertices.")
            with np.load(staging / "psr_my.npz", allow_pickle=False) as archive:
                if set(archive.files) != {"psr"} or archive["psr"].shape != (args.psr_resolution,) * 3:
                    raise PreparationError("بنية psr_my.npz غير متوافقة مع قارئ MADCrowner.")
                if not np.all(np.isfinite(archive["psr"])):
                    raise PreparationError("حقل PSR الناتج يحتوي قيمًا غير منتهية.")

            report = {
                "status": "candidate_generated_requires_training_loader_and_visual_review",
                "source_file_names_recorded": False,
                "tooth_number": 46,
                "margin_point_count": int(len(margin)),
                "units_per_mm_assumption": float(args.units_per_mm),
                "coordinate_choices": {
                    "margin": "raw XML coordinates",
                    "crown": "raw crown STL; no ZRotationMatrix applied",
                    "jaw": "selected ScanFile TransformationMatrix transpose_forward",
                },
                "surface_checks": {
                    "margin_to_crown_surface_max_mm": float(crown_distance["max_mm"]),
                    "margin_to_jaw_surface_max_mm": float(jaw_distance["max_mm"]),
                    "crown_boundary_edge_count": crown_boundary_edge_count,
                    "note": "A small distance to the whole crown surface does not identify the clinical margin edge.",
                },
                "crop_size_mm": float(args.crop_size_mm),
                "crown_vertex_count": int(len(crown.vertices)),
                "jaw_cropped_vertex_count": int(len(pna_crop.vertices)),
                "jaw_cropped_face_count": int(len(pna_crop.faces)),
                "jaw_crop_source_vertex_count": int(len(pna_original_indices)),
                "boundary_labels": pna_label_report,
                "crown_margin_labels": crown_label_report,
                "curvatures": {
                    "method": "VTK vtkCurvatures mean; raw output, no unverified normalization",
                    "count": int(len(curvatures)),
                    "min": float(np.min(curvatures)),
                    "max": float(np.max(curvatures)),
                },
                "psr": {
                    "key": "psr",
                    "shape": [int(args.psr_resolution)] * 3,
                    "dtype": "float16",
                    "method": "NumPy CPU implementation of SAP DPSR conventions (sig=0, shift=True, scale=True)",
                    "scale_mode": psr_scale_mode,
                },
                "training_readiness": {
                    "approved_for_training": False,
                    "reasons": [
                        "Margin-to-crown distance is to the whole surface, not proof of the lower edge.",
                        "Margin vertex labels use an explicit nearest-vertex approximation and need visual review.",
                        "The upstream loader independently normalizes crown and PNA point sets; integration behavior must be reviewed before training.",
                    ],
                },
                "outputs": ["crown.ply", "pna_crop.ply", "boundary_labels.npy", "crown_attributes.h5", "psr_my.npz"],
            }
            (staging / "prep_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        os.replace(staging, output_path)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="تجهيز عينة تدريب واحدة محليًا؛ الملفات لا تُرفع إلى أي خدمة."
    )
    parser.add_argument("--case", required=True, help="مجلد حالة مستخرج أو ملف RAR محلي")
    parser.add_argument("--out", required=True, help="مجلد عينة جديد خارج المستودع ومجلد الحالة")
    parser.add_argument("--constructioninfo", help="اسم XML إذا كان داخل الحالة أكثر من ملف")
    parser.add_argument("--crown", help="اسم تاج STL إذا لم يطابق اسم السن 46 الافتراضي")
    parser.add_argument("--modeldie", help="اسم modeldie STL إذا لم يطابق اسم السن 46 الافتراضي")
    parser.add_argument("--jaw-hint", default="mandibular", help="جزء من اسم مرجع مسح الفك")
    parser.add_argument("--units-per-mm", type=float, default=1.0)
    parser.add_argument("--crop-size-mm", type=float, default=20.0)
    parser.add_argument("--margin-tolerance-mm", type=float, default=0.15)
    parser.add_argument("--max-snap-mm", type=float, default=1.5,
                        help="أقصى مسافة مسموحة لإسناد نقاط XML إلى رؤوس الفك")
    parser.add_argument("--max-crown-snap-mm", type=float, default=0.75)
    parser.add_argument("--max-jaw-margin-mm", type=float, default=1.5,
                        help="حد توقف لقياس السطح فقط؛ ليس معيار قبول سريري")
    parser.add_argument("--max-crown-margin-mm", type=float, default=0.1)
    parser.add_argument("--psr-resolution", type=int, default=128)
    parser.add_argument("--verbose", action="store_true", help="طباعة تفاصيل خطأ محلية؛ قد تتضمن مسارات ملفات")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_sample(args)
    except Exception as exc:
        if args.verbose:
            print(f"فشل تجهيز العينة ({type(exc).__name__}): {exc}", file=sys.stderr)
        else:
            print(
                f"توقف تجهيز العينة ({type(exc).__name__}). لم تُحفظ عينة نهائية؛ "
                "لا ترسل ملفات الحالة أو التقرير الكامل.",
                file=sys.stderr,
            )
        return 1
    print("تم إنشاء ملفات العينة الخمسة محليًا مع prep_report.json.")
    print(f"حالة التحقق: {report['status']} — لا تبدأ التدريب قبل مراجعة المخرجات ومواءمة DataLoader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
