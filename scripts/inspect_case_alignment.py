#!/usr/bin/env python3
"""Local first-pass inspection of one exocad case; does not upload case data.

The XML schema and transform direction have not yet been verified on a real case.
This script therefore exports raw and candidate-transformed meshes separately and
reports margin-to-prep surface distances as diagnostics, not as proof of alignment.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np
import rarfile
import trimesh

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
FILE_RE = re.compile(r"[^\r\n\t<>\"]*?\.(?:ply|stl)", re.IGNORECASE)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1].strip().lower()


def descendants_named(root: ET.Element, name: str) -> list[ET.Element]:
    wanted = name.lower()
    return [el for el in root.iter() if local_name(el.tag) == wanted]


def text_of(el: ET.Element) -> str:
    return " ".join(part.strip() for part in el.itertext() if part and part.strip())


def parse_case_xml(xml_path: Path) -> dict:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"تعذّر قراءة XML: {exc}") from exc

    type_elements = descendants_named(root, "ReconstructionType")
    types = [text_of(el) for el in type_elements]
    if len(types) != 1 or types[0].casefold() != "anatomiccrown":
        raise ValueError(
            "أحتاج قيمة ReconstructionType واحدة تساوي AnatomicCrown؛ "
            f"القيم التي وجدتها: {types or 'لا يوجد حقل'}"
        )

    number_elements = descendants_named(root, "Number")
    number_values = [re.findall(r"\d+", text_of(el)) for el in number_elements]
    if len(number_values) != 1 or number_values[0] != ["46"]:
        raise ValueError(
            "أحتاج عنصر <Number> واحدًا وقيمته 46 حتى لا نؤكد رقم سن من حقل آخر بالخطأ؛ "
            f"القيم المكتشفة: {number_values or 'لا يوجد حقل'}"
        )

    scan_refs: list[str] = []
    for group in descendants_named(root, "ScanFiles"):
        for el in group.iter():
            candidates = [text_of(el), *[str(v) for v in el.attrib.values()]]
            for candidate in candidates:
                for match in FILE_RE.findall(candidate):
                    value = match.strip().strip("' ").replace("\\", "/")
                    if value and value not in scan_refs:
                        scan_refs.append(value)

    margin_groups = descendants_named(root, "Margin")
    vec_elements = [vec for group in margin_groups for vec in group.iter()
                    if local_name(vec.tag) == "vec3"]
    margin_points: list[list[float]] = []
    for vec in vec_elements:
        xyz: dict[str, float] = {}
        for node in vec.iter():
            key = local_name(node.tag)
            if key in {"x", "y", "z"}:
                vals = NUMBER_RE.findall(text_of(node))
                if vals:
                    xyz[key] = float(vals[0])
        if all(axis in xyz for axis in ("x", "y", "z")):
            point = [xyz["x"], xyz["y"], xyz["z"]]
        else:
            vals = NUMBER_RE.findall(text_of(vec))
            if len(vals) != 3:
                raise ValueError(f"تعذّر تفسير Vec3 كإحداثيات X,Y,Z: {text_of(vec)!r}")
            point = [float(v) for v in vals]
        margin_points.append(point)
    if not margin_points:
        raise ValueError("لم أجد نقاط Vec3 داخل <Margin>.")

    matrices = extract_matrices(root)
    return {
        "reconstruction_types": types,
        "number_values": number_values,
        "scan_refs": scan_refs,
        "margin_points": np.asarray(margin_points, dtype=np.float64),
        "matrices": matrices,
    }


def _matrix_from_element(el: ET.Element) -> np.ndarray | None:
    # Prefer one continuous 16-value sequence; do not guess around extra values.
    text_vals = NUMBER_RE.findall(text_of(el))
    candidates = [text_vals]
    candidates.extend(NUMBER_RE.findall(str(value)) for value in el.attrib.values())
    for vals in candidates:
        if len(vals) == 16:
            matrix = np.asarray([float(v) for v in vals], dtype=np.float64).reshape(4, 4)
            if np.isfinite(matrix).all():
                return matrix
    return None


def extract_matrices(root: ET.Element) -> list[dict]:
    containers = descendants_named(root, "MatrixToScanDataFiles")
    candidates: list[tuple[ET.Element, np.ndarray]] = []
    for container in containers:
        matrix_nodes = [el for el in container.iter() if el is not container
                        and "matrix" in local_name(el.tag)]
        if not matrix_nodes:
            matrix_nodes = [container]
        for node in matrix_nodes:
            matrix = _matrix_from_element(node)
            if matrix is not None:
                candidates.append((node, matrix))

    # Some XML variants put TransformationMatrix outside the named container.
    if not candidates:
        for node in descendants_named(root, "TransformationMatrix"):
            matrix = _matrix_from_element(node)
            if matrix is not None:
                candidates.append((node, matrix))

    result: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for node, matrix in candidates:
        key = tuple(np.round(matrix, 12).ravel().tolist())
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "matrix": matrix,
            "xml_tag": local_name(node.tag),
            "xml_attributes": {str(k): str(v) for k, v in node.attrib.items()},
            "xml_text_hint": text_of(node)[:160],
        })
    return result


def normalize_member_name(name: str) -> str:
    return name.replace("\\", "/").rstrip("/")


def member_basename(name: str) -> str:
    return PurePosixPath(normalize_member_name(name)).name.casefold()


def choose_xml_path(files: list[Path], requested: str | None) -> Path:
    candidates = [p for p in files if p.suffix.casefold() == ".constructioninfo"]
    if requested:
        candidates = [p for p in candidates if p.name.casefold() == Path(requested).name.casefold()
                      or str(p).casefold() == requested.casefold()]
    if len(candidates) != 1:
        raise ValueError(f"أحتاج ملف constructionInfo واحدًا بالضبط؛ وجدت {len(candidates)}.")
    return candidates[0]


def select_stl(files: list[Path], pattern: str | None, tooth_suffix: str) -> Path:
    expected = (pattern or tooth_suffix).casefold()
    matches = [p for p in files if p.suffix.casefold() == ".stl"
               and (p.name.casefold() == expected or p.name.casefold().endswith(expected))]
    if len(matches) != 1:
        raise ValueError(f"أحتاج ملف STL واحدًا يطابق {expected!r}؛ وجدت {len(matches)}.")
    return matches[0]


def select_scan_files(files: list[Path], scan_refs: list[str]) -> tuple[list[Path], str | None]:
    ply_files = [p for p in files if p.suffix.casefold() == ".ply"]
    if not scan_refs:
        return ply_files, "لم تُستخرج أسماء من ScanFiles؛ أُدرجت ملفات PLY الموجودة كلها كمرشحات."
    wanted = {member_basename(name) for name in scan_refs}
    matches = [p for p in ply_files if p.name.casefold() in wanted]
    if not matches:
        return ply_files, "أسماء ScanFiles لم تطابق أسماء PLY؛ أُدرجت ملفات PLY الموجودة كلها كمرشحات."
    return matches, None


def extract_rar_member(archive_path: Path, member_name: str, destination: Path) -> Path:
    """Extract one selected member through rarfile's supported local backend."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rarfile.RarFile(str(archive_path)) as archive:
            with archive.open(member_name) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    except Exception as exc:
        raise RuntimeError(
            "تعذّر فك ملف من RAR. ثبّت 7-Zip أو UnRAR وأضف أداة فك RAR إلى PATH، "
            "ثم افتح PowerShell من جديد. إذا كان الملف متعدد الأجزاء، اترك كل الأجزاء "
            "في المجلد نفسه. السبب: " + str(exc)
        ) from exc
    return destination


def resolve_input(input_path: Path, work_dir: Path, construction_name: str | None,
                  crown_name: str | None, modeldie_name: str | None,
                  scan_refs_from_xml: bool = False) -> tuple[Path, list[Path], Path, Path, str | None]:
    """Return XML, scans, crown, die, and optional selection warning."""
    if input_path.is_dir():
        all_files = [p for p in input_path.rglob("*") if p.is_file()]
        xml_path = choose_xml_path(all_files, construction_name)
        parsed = parse_case_xml(xml_path)
        scans, warning = select_scan_files(all_files, parsed["scan_refs"])
        if not scans:
            raise ValueError("لم أجد ملفات PLY للمسح.")
        crown = select_stl(all_files, crown_name, "46-crown_cad.stl")
        die = select_stl(all_files, modeldie_name, "46-modeldie.stl")
        return xml_path, scans, crown, die, warning

    if input_path.is_file() and input_path.suffix.casefold() == ".rar":
        try:
            with rarfile.RarFile(str(input_path)) as archive:
                members = [info.filename for info in archive.infolist() if not info.isdir()]
        except Exception as exc:
            raise RuntimeError(f"تعذّرت قراءة فهرس RAR: {exc}") from exc
        xml_members = [m for m in members if m.casefold().endswith(".constructioninfo")]
        if construction_name:
            xml_members = [m for m in xml_members
                           if member_basename(m) == Path(construction_name).name.casefold()
                           or normalize_member_name(m).casefold() == normalize_member_name(construction_name).casefold()]
        if len(xml_members) != 1:
            raise ValueError(f"أحتاج ملف constructionInfo واحدًا داخل RAR؛ وجدت {len(xml_members)}.")
        xml_path = extract_rar_member(input_path, xml_members[0], work_dir / "case.constructionInfo")
        parsed = parse_case_xml(xml_path)

        wanted_scan_names = {member_basename(name) for name in parsed["scan_refs"]}
        scan_members = [m for m in members if member_basename(m).endswith(".ply")
                        and (not wanted_scan_names or member_basename(m) in wanted_scan_names)]
        warning = None
        if not parsed["scan_refs"]:
            warning = "لم تُستخرج أسماء من ScanFiles؛ أُدرجت كل ملفات PLY في RAR كمرشحات."
            scan_members = [m for m in members if member_basename(m).endswith(".ply")]
        elif not scan_members:
            warning = "أسماء ScanFiles لم تطابق أسماء PLY؛ أُدرجت كل ملفات PLY في RAR كمرشحات."
            scan_members = [m for m in members if member_basename(m).endswith(".ply")]
        scans = [extract_rar_member(input_path, name, work_dir / f"scan_{i:02d}.ply")
                 for i, name in enumerate(scan_members)]

        crown_expected = (crown_name or "46-crown_cad.stl").casefold()
        die_expected = (modeldie_name or "46-modeldie.stl").casefold()
        crown_members = [m for m in members if member_basename(m) == crown_expected
                         or member_basename(m).endswith(crown_expected)]
        die_members = [m for m in members if member_basename(m) == die_expected
                       or member_basename(m).endswith(die_expected)]
        if len(crown_members) != 1:
            raise ValueError(f"أحتاج تاج STL واحدًا يطابق {crown_expected!r} داخل RAR؛ وجدت {len(crown_members)}.")
        if len(die_members) != 1:
            raise ValueError(f"أحتاج modeldie STL واحدًا يطابق {die_expected!r} داخل RAR؛ وجدت {len(die_members)}.")
        crown = extract_rar_member(input_path, crown_members[0], work_dir / "reference_crown.stl")
        die = extract_rar_member(input_path, die_members[0], work_dir / "prepared_die.stl")
        return xml_path, scans, crown, die, warning

    raise ValueError("--case لازم يكون مسار مجلد حالة أو ملف .rar واحد.")


def load_geometry(path: Path):
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.dump(concatenate=True)
    if not isinstance(geometry, (trimesh.Trimesh, trimesh.points.PointCloud)):
        raise ValueError(f"نوع هندسة غير مدعوم في {path.name}: {type(geometry).__name__}")
    if len(geometry.vertices) == 0:
        raise ValueError(f"الملف فارغ: {path.name}")
    return geometry


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    result = hom @ matrix.T
    w = result[:, 3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("المصفوفة أعطت إحداثيات متجانسة غير صالحة.")
    return result[:, :3] / w[:, None]


def matrix_variants(matrix: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    for layout, base in (("row", matrix), ("transpose", matrix.T)):
        variants.append((f"{layout}_forward", base))
        try:
            variants.append((f"{layout}_inverse", np.linalg.inv(base)))
        except np.linalg.LinAlgError:
            pass
    return variants


def margin_surface_distances(mesh: trimesh.Trimesh, points: np.ndarray,
                             units_per_mm: float) -> dict:
    # closest_point measures to triangle surfaces, not merely to mesh vertices.
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    distances_mm = np.asarray(distances, dtype=np.float64) / units_per_mm
    return {
        "point_count": int(len(distances_mm)),
        "units_per_mm_assumption": float(units_per_mm),
        "max_mm": float(np.max(distances_mm)),
        "median_mm": float(np.median(distances_mm)),
        "p95_mm": float(np.percentile(distances_mm, 95)),
        "fraction_at_or_below_threshold": float(np.mean(distances_mm <= 0.5)),
        "all_points_at_or_below_0_5mm": bool(np.all(distances_mm <= 0.5)),
    }


def export_variant(geometry, transform: np.ndarray, output_path: Path) -> None:
    copy = geometry.copy()
    copy.apply_transform(transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copy.export(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="يفحص حالة واحدة محليًا؛ لا يرفع أي ملف ولا يثبت أن المحاذاة صحيحة 100%."
    )
    parser.add_argument("--case", required=True, help="مسار مجلد الحالة أو ملف .rar")
    parser.add_argument("--out", required=True, help="مجلد نتائج الاختبار المحلي")
    parser.add_argument("--constructioninfo", help="اسم ملف XML إذا وُجد أكثر من واحد")
    parser.add_argument("--crown", help="اسم تاج STL إذا لم يكن اسمه 46-crown_cad.stl")
    parser.add_argument("--modeldie", help="اسم modeldie STL إذا لم يكن اسمه 46-modeldie.stl")
    parser.add_argument("--units-per-mm", type=float, default=1.0,
                        help="عدد وحدات الإحداثيات في المليمتر؛ الافتراضي 1 ويعني افتراض أن الإحداثيات mm")
    args = parser.parse_args()

    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    if args.units_per_mm <= 0:
        parser.error("--units-per-mm يجب أن يكون أكبر من صفر.")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="dental_case_inspect_") as temp_name:
            work_dir = Path(temp_name)
            xml_path, scan_paths, crown_path, die_path, selection_warning = resolve_input(
                input_path, work_dir, args.constructioninfo, args.crown, args.modeldie
            )
            parsed = parse_case_xml(xml_path)
            margins = parsed["margin_points"]
            die_geometry = load_geometry(die_path)
            if not isinstance(die_geometry, trimesh.Trimesh) or len(die_geometry.faces) == 0:
                raise ValueError("ملف modeldie لازم يحتوي سطحًا مثلثيًا، وليس نقاطًا فقط.")

            # Save local, anonymized copies of the two reference STL files.
            shutil.copy2(crown_path, output_dir / "reference_crown_original.stl")
            shutil.copy2(die_path, output_dir / "prepared_die_original.stl")
            np.save(output_dir / "margin_points_raw.npy", margins)

            matrix_reports: list[dict] = []
            distance_reports: list[dict] = []
            for matrix_index, item in enumerate(parsed["matrices"]):
                matrix = item["matrix"]
                variants = matrix_variants(matrix)
                matrix_dir = output_dir / f"matrix_{matrix_index:02d}"
                matrix_dir.mkdir(parents=True, exist_ok=True)
                for scan_index, scan_path in enumerate(scan_paths):
                    geometry = load_geometry(scan_path)
                    shutil.copy2(scan_path, matrix_dir / f"scan_{scan_index:02d}_raw.ply")
                    for variant_name, transform in variants:
                        export_variant(
                            geometry, transform,
                            matrix_dir / f"scan_{scan_index:02d}_{variant_name}.ply"
                        )
                variant_distances = {}
                for variant_name, transform in variants:
                    transformed_margin = apply_transform(margins, transform)
                    np.save(matrix_dir / f"margin_{variant_name}.npy", transformed_margin)
                    try:
                        variant_distances[variant_name] = margin_surface_distances(
                            die_geometry, transformed_margin, args.units_per_mm
                        )
                    except Exception as exc:
                        variant_distances[variant_name] = {"error": str(exc)}
                try:
                    raw_distance = margin_surface_distances(die_geometry, margins, args.units_per_mm)
                except Exception as exc:
                    raw_distance = {"error": str(exc)}
                distance_reports.append({
                    "matrix_index": matrix_index,
                    "assumption": "مقارنة margin بعد التحويل مع modeldie الخام؛ صالحة فقط إذا كان modeldie في إطار المسح نفسه",
                    "raw_margin_vs_raw_modeldie": raw_distance,
                    "transformed_margin_vs_raw_modeldie": variant_distances,
                })
                matrix_reports.append({
                    "index": matrix_index,
                    "xml_tag": item["xml_tag"],
                    "row_interpretation_affine_check": bool(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)),
                    "transpose_interpretation_affine_check": bool(np.allclose(matrix.T[3], [0, 0, 0, 1], atol=1e-5)),
                    "variants_exported": [name for name, _ in variants],
                })

            if not parsed["matrices"]:
                for scan_index, scan_path in enumerate(scan_paths):
                    shutil.copy2(scan_path, output_dir / f"scan_{scan_index:02d}_raw.ply")
                try:
                    distance_reports.append({
                        "matrix_index": None,
                        "raw_margin_vs_raw_modeldie": margin_surface_distances(
                            die_geometry, margins, args.units_per_mm
                        ),
                    })
                except Exception as exc:
                    distance_reports.append({"matrix_index": None, "error": str(exc)})

            report = {
                "case_check": "PASS: AnatomicCrown and tooth 46 found",
                "scan_reference_count_in_xml": len(parsed["scan_refs"]),
                "scan_candidates_processed": len(scan_paths),
                "margin_point_count": int(len(margins)),
                "matrix_candidate_count": len(parsed["matrices"]),
                "distance_threshold_mm": 0.5,
                "units_per_mm_assumption": args.units_per_mm,
                "selection_note": selection_warning,
                "matrix_candidates": matrix_reports,
                "margin_to_modeldie_distance_diagnostics": distance_reports,
                "warning": (
                    "هذه نتائج تشخيصية وليست إثباتًا لمحاذاة 100%. يجب تأكيد اتجاه المصفوفة، "
                    "وحدات القياس، وإطار modeldie بصريًا على الحالة في عارض Mesh/exocad. "
                    "تفسير المسافة يفترض أن modeldie الخام في إطار المسح نفسه."
                ),
            }
            report_path = output_dir / "diagnostic_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"\nحُفظت النتائج محليًا في: {output_dir}")
            return 0
    except Exception as exc:
        print(f"فشل الفحص: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
