#!/usr/bin/env python3
"""Local first-pass inspection of one exocad case; case data stays on disk.

Schema was checked against one real .constructionInfo file. Transform direction,
units, and the relationship between construction matrices and mesh files still
need visual confirmation, so candidate transforms are exported separately.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np
import rarfile
import trimesh

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
MATRIX_FIELD_RE = re.compile(r"^_(\d)(\d)$")
FILE_RE = re.compile(r'[^\r\n\t<>"]*?\.(?:ply|stl)', re.IGNORECASE)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1].strip().lower()


def named(root: ET.Element, name: str) -> list[ET.Element]:
    wanted = name.casefold()
    return [el for el in root.iter() if local_name(el.tag) == wanted]


def direct_named(root: ET.Element, name: str) -> list[ET.Element]:
    wanted = name.casefold()
    return [el for el in list(root) if local_name(el.tag) == wanted]


def text_of(el: ET.Element) -> str:
    return " ".join(part.strip() for part in el.itertext() if part and part.strip())


def matrix_from_xml(el: ET.Element) -> np.ndarray | None:
    """Read exocad's named _00 ... _33 fields, with a strict 16-value fallback."""
    fields: dict[tuple[int, int], float] = {}
    for child in list(el):
        match = MATRIX_FIELD_RE.fullmatch(local_name(child.tag))
        if not match:
            continue
        values = NUMBER_RE.findall(text_of(child))
        if len(values) != 1:
            return None
        key = (int(match.group(1)), int(match.group(2)))
        if key in fields:
            return None
        fields[key] = float(values[0])
    if len(fields) == 16:
        return np.asarray([[fields[(r, c)] for c in range(4)] for r in range(4)], dtype=np.float64)

    values = NUMBER_RE.findall(text_of(el))
    if len(values) == 16:
        return np.asarray([float(v) for v in values], dtype=np.float64).reshape(4, 4)
    return None


def matrix_variants(matrix: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Export both matrix-layout and direction hypotheses; none is pre-certified."""
    variants: list[tuple[str, np.ndarray]] = []
    for layout, base in (("as_tagged", matrix), ("transpose", matrix.T)):
        variants.append((f"{layout}_forward", base))
        try:
            variants.append((f"{layout}_inverse", np.linalg.inv(base)))
        except np.linalg.LinAlgError:
            pass
    return variants


def parse_vec3(vec: ET.Element) -> list[float]:
    xyz: dict[str, float] = {}
    for node in vec.iter():
        axis = local_name(node.tag)
        if axis in {"x", "y", "z"}:
            values = NUMBER_RE.findall(text_of(node))
            if len(values) == 1:
                xyz[axis] = float(values[0])
    if all(axis in xyz for axis in ("x", "y", "z")):
        return [xyz["x"], xyz["y"], xyz["z"]]
    values = NUMBER_RE.findall(text_of(vec))
    if len(values) == 3:
        return [float(v) for v in values]
    raise ValueError("صيغة Vec3 غير معروفة؛ توقفت بدل تخمين إحداثيات الـ margin.")


def parse_case_xml(xml_path: Path) -> dict:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"تعذّر قراءة XML: {exc}") from exc

    # The real exocad file uses this misspelling in the root tag.
    if local_name(root.tag) != "constuctioninfo":
        raise ValueError(
            f"وسم الجذر غير متوقع: {local_name(root.tag)!r}; المتوقع ConstuctionInfo."
        )

    types = [text_of(el) for el in named(root, "ReconstructionType")]
    if len(types) != 1 or types[0].casefold() != "anatomiccrown":
        raise ValueError(f"أحتاج ReconstructionType واحدًا = AnatomicCrown؛ الموجود: {types or 'لا يوجد'}")

    numbers = [re.findall(r"\d+", text_of(el)) for el in named(root, "Number")]
    if len(numbers) != 1 or numbers[0] != ["46"]:
        raise ValueError(f"أحتاج عنصر Number واحدًا وقيمته 46؛ القيم المكتشفة: {numbers or 'لا يوجد'}")

    margin_groups = named(root, "Margin")
    if len(margin_groups) != 1:
        raise ValueError(f"أحتاج Margin واحدًا؛ وجدت {len(margin_groups)}.")
    margin_points = [parse_vec3(el) for el in margin_groups[0].iter()
                     if local_name(el.tag) == "vec3"]
    if not margin_points:
        raise ValueError("لم أجد Vec3 داخل Margin.")

    # In the actual file each ScanFile has its own FileName and TransformationMatrix.
    scan_groups = named(root, "ScanFiles")
    if len(scan_groups) != 1:
        raise ValueError(f"أحتاج ScanFiles واحدًا؛ وجدت {len(scan_groups)}.")
    scan_records: list[dict] = []
    for index, scan_el in enumerate(direct_named(scan_groups[0], "ScanFile")):
        file_names = direct_named(scan_el, "FileName")
        if len(file_names) != 1:
            raise ValueError(f"ScanFile رقم {index + 1} لا يحتوي FileName واحدًا.")
        filename_text = text_of(file_names[0])
        file_matches = FILE_RE.findall(filename_text)
        if len(file_matches) != 1:
            raise ValueError(f"تعذّر استخراج اسم ملف واحد من FileName رقم {index + 1}.")
        transform_nodes = direct_named(scan_el, "TransformationMatrix")
        scan_matrix = matrix_from_xml(transform_nodes[0]) if len(transform_nodes) == 1 else None
        scan_records.append({
            "file_ref": file_matches[0].strip().replace("\\", "/"),
            "matrix": scan_matrix,
            "matrix_present": scan_matrix is not None,
        })
    if not scan_records:
        raise ValueError("لم أجد عناصر ScanFile داخل ScanFiles.")

    # These matrices occur at the root and under ConstructionFile in the supplied XML.
    construction_matrices: list[dict] = []
    for matrix_el in named(root, "MatrixToScanDataFiles"):
        matrix = matrix_from_xml(matrix_el)
        if matrix is None:
            raise ValueError("وجدت MatrixToScanDataFiles لكن لم أستطع قراءة حقول _00 إلى _33.")
        # Keep only tag names in scope metadata, never patient values or filenames.
        ancestors = []
        parent_map = {child: parent for parent in root.iter() for child in parent}
        parent = parent_map.get(matrix_el)
        while parent is not None:
            ancestors.append(local_name(parent.tag))
            parent = parent_map.get(parent)
        scope = "construction_file" if "constructionfile" in ancestors else "root"
        construction_matrices.append({"matrix": matrix, "scope": scope})

    return {
        "margin_points": np.asarray(margin_points, dtype=np.float64),
        "scan_records": scan_records,
        "construction_matrices": construction_matrices,
    }


def normalize_member_name(name: str) -> str:
    return name.replace("\\", "/").rstrip("/")


def member_basename(name: str) -> str:
    return PurePosixPath(normalize_member_name(name)).name.casefold()


def match_path_by_ref(files: list[Path], reference: str) -> Path:
    ref_norm = normalize_member_name(reference).casefold()
    ref_base = member_basename(reference)
    candidates = [p for p in files if p.suffix.casefold() == ".ply"
                  and (p.name.casefold() == ref_base or str(p).replace("\\", "/").casefold().endswith(ref_norm))]
    if len(candidates) != 1:
        raise ValueError(f"اسم مسح من ScanFiles لم يطابق ملف PLY واحدًا بالضبط؛ عدد المطابقات: {len(candidates)}.")
    return candidates[0]


def choose_one(files: list[Path], suffix: str, expected: str, custom: str | None = None) -> Path:
    wanted = (custom or expected).casefold()
    candidates = [p for p in files if p.suffix.casefold() == suffix.casefold()
                  and (p.name.casefold() == wanted or p.name.casefold().endswith(wanted))]
    if len(candidates) != 1:
        raise ValueError(f"أحتاج ملفًا واحدًا يطابق {wanted!r}؛ عدد المطابقات: {len(candidates)}.")
    return candidates[0]


def extract_rar_member(archive_path: Path, member_name: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rarfile.RarFile(str(archive_path)) as archive:
            with archive.open(member_name) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    except Exception as exc:
        raise RuntimeError(
            "تعذّر فك ملف من RAR. تأكد من تثبيت 7-Zip أو UnRAR وإضافته إلى PATH، "
            "وضع كل أجزاء الأرشيف متعدد الأجزاء في المجلد نفسه. السبب: " + str(exc)
        ) from exc
    return destination


def resolve_folder(input_path: Path, work_dir: Path, construction_name: str | None,
                   crown_name: str | None, modeldie_name: str | None) -> tuple:
    files = [p for p in input_path.rglob("*") if p.is_file()]
    xmls = [p for p in files if p.suffix.casefold() == ".constructioninfo"]
    if construction_name:
        xmls = [p for p in xmls if p.name.casefold() == Path(construction_name).name.casefold()]
    if len(xmls) != 1:
        raise ValueError(f"أحتاج ملف constructionInfo واحدًا؛ وجدت {len(xmls)}.")
    parsed = parse_case_xml(xmls[0])
    scans = []
    for record in parsed["scan_records"]:
        scans.append({"path": match_path_by_ref(files, record["file_ref"]), "record": record})
    crown = choose_one(files, ".stl", "46-crown_cad.stl", crown_name)
    die = choose_one(files, ".stl", "46-modeldie.stl", modeldie_name)
    return xmls[0], parsed, scans, crown, die


def resolve_rar(input_path: Path, work_dir: Path, construction_name: str | None,
                crown_name: str | None, modeldie_name: str | None) -> tuple:
    try:
        with rarfile.RarFile(str(input_path)) as archive:
            members = [info.filename for info in archive.infolist() if not info.isdir()]
    except Exception as exc:
        raise RuntimeError(f"تعذّر قراءة فهرس RAR: {exc}") from exc

    xml_members = [m for m in members if m.casefold().endswith(".constructioninfo")]
    if construction_name:
        xml_members = [m for m in xml_members if member_basename(m) == Path(construction_name).name.casefold()]
    if len(xml_members) != 1:
        raise ValueError(f"أحتاج ملف constructionInfo واحدًا داخل RAR؛ وجدت {len(xml_members)}.")
    xml_path = extract_rar_member(input_path, xml_members[0], work_dir / "case.constructionInfo")
    parsed = parse_case_xml(xml_path)

    scans = []
    for index, record in enumerate(parsed["scan_records"]):
        ref_base = member_basename(record["file_ref"])
        matches = [m for m in members if m.casefold().endswith(".ply") and member_basename(m) == ref_base]
        if len(matches) != 1:
            raise ValueError(f"اسم مسح من ScanFiles لم يطابق ملف PLY واحدًا داخل RAR؛ عدد المطابقات: {len(matches)}.")
        path = extract_rar_member(input_path, matches[0], work_dir / f"scan_{index:02d}.ply")
        scans.append({"path": path, "record": record})

    crown = choose_rar_member(members, ".stl", "46-crown_cad.stl", crown_name,
                              input_path, work_dir / "reference_crown.stl")
    die = choose_rar_member(members, ".stl", "46-modeldie.stl", modeldie_name,
                            input_path, work_dir / "prepared_die.stl")
    return xml_path, parsed, scans, crown, die


def choose_rar_member(members: list[str], suffix: str, expected: str, custom: str | None,
                      archive_path: Path, destination: Path) -> Path:
    wanted = (custom or expected).casefold()
    matches = [m for m in members if m.casefold().endswith(suffix.casefold())
               and (member_basename(m) == wanted or member_basename(m).endswith(wanted))]
    if len(matches) != 1:
        raise ValueError(f"أحتاج ملفًا واحدًا يطابق {wanted!r} داخل RAR؛ عدد المطابقات: {len(matches)}.")
    return extract_rar_member(archive_path, matches[0], destination)


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    result = hom @ matrix.T
    w = result[:, 3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("المصفوفة أعطت إحداثيات غير صالحة.")
    return result[:, :3] / w[:, None]


def load_geometry(path: Path):
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.dump(concatenate=True)
    if not isinstance(geometry, (trimesh.Trimesh, trimesh.points.PointCloud)) or len(geometry.vertices) == 0:
        raise ValueError(f"نوع أو محتوى Mesh غير مدعوم في {path.suffix}.")
    return geometry


def export_variant(geometry, matrix: np.ndarray, destination: Path) -> None:
    transformed = geometry.copy()
    transformed.apply_transform(matrix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transformed.export(str(destination))


def surface_distance_report(mesh: trimesh.Trimesh, points: np.ndarray, units_per_mm: float) -> dict:
    # Exact nearest distance to triangle surfaces (rtree is required by trimesh).
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    distances_mm = np.asarray(distances, dtype=np.float64) / units_per_mm
    return {
        "point_count": int(len(distances_mm)),
        "max_mm": float(np.max(distances_mm)),
        "median_mm": float(np.median(distances_mm)),
        "p95_mm": float(np.percentile(distances_mm, 95)),
        "fraction_at_or_below_0_5mm": float(np.mean(distances_mm <= 0.5)),
        "all_points_at_or_below_0_5mm": bool(np.all(distances_mm <= 0.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="فحص محلي لحالة exocad واحدة؛ لا يرفع الملفات ولا يثبت المحاذاة 100%.")
    parser.add_argument("--case", required=True, help="مسار مجلد الحالة أو ملف .rar")
    parser.add_argument("--out", required=True, help="مجلد نتائج الاختبار المحلي، خارج مجلد الحالة")
    parser.add_argument("--constructioninfo", help="اسم XML إذا احتوى المجلد/الأرشيف أكثر من ملف")
    parser.add_argument("--crown", help="اسم تاج STL إذا لم يطابق 46-crown_cad.stl")
    parser.add_argument("--modeldie", help="اسم modeldie STL إذا لم يطابق 46-modeldie.stl")
    parser.add_argument("--units-per-mm", type=float, default=1.0,
                        help="عدد وحدات الإحداثيات في المليمتر؛ 1 يعني افتراض أن البيانات بالـ mm")
    args = parser.parse_args()

    case_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    if args.units_per_mm <= 0:
        parser.error("--units-per-mm يجب أن يكون أكبر من صفر.")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="dental_case_inspect_") as temp:
            work_dir = Path(temp)
            if case_path.is_dir():
                xml_path, parsed, scans, crown, die = resolve_folder(
                    case_path, work_dir, args.constructioninfo, args.crown, args.modeldie
                )
            elif case_path.is_file() and case_path.suffix.casefold() == ".rar":
                xml_path, parsed, scans, crown, die = resolve_rar(
                    case_path, work_dir, args.constructioninfo, args.crown, args.modeldie
                )
            else:
                raise ValueError("--case يجب أن يكون مسار مجلد حالة أو ملف .rar واحدًا.")

            margins = parsed["margin_points"]
            die_mesh = load_geometry(die)
            if not isinstance(die_mesh, trimesh.Trimesh) or len(die_mesh.faces) == 0:
                raise ValueError("modeldie لازم يحتوي سطحًا مثلثيًا.")
            shutil.copy2(crown, output_dir / "reference_crown_original.stl")
            shutil.copy2(die, output_dir / "prepared_die_original.stl")
            np.save(output_dir / "margin_points_raw.npy", margins)

            scan_reports = []
            for scan_index, scan_item in enumerate(scans):
                scan_path = scan_item["path"]
                record = scan_item["record"]
                geometry = load_geometry(scan_path)
                scan_dir = output_dir / f"scan_{scan_index:02d}"
                scan_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(scan_path, scan_dir / "raw.ply")
                variants_report = []
                if record["matrix"] is not None:
                    for variant_name, matrix in matrix_variants(record["matrix"]):
                        export_variant(geometry, matrix, scan_dir / f"scanfile_{variant_name}.ply")
                        variants_report.append(variant_name)
                scan_reports.append({
                    "scan_index": scan_index,
                    "transformation_matrix_found": record["matrix"] is not None,
                    "candidate_meshes_written": variants_report,
                })

            distance_reports = []
            try:
                distance_reports.append({
                    "matrix_source": "raw_margin_vs_raw_modeldie",
                    "assumption": "التاج المحفور والـ margin الخام في الإطار نفسه",
                    "result": surface_distance_report(die_mesh, margins, args.units_per_mm),
                })
            except Exception as exc:
                distance_reports.append({"matrix_source": "raw_margin_vs_raw_modeldie", "error": str(exc)})

            # MatrixToScanDataFiles is distinct from each ScanFile/TransformationMatrix.
            # Test it on margin candidates; do not silently apply it to PLY scans.
            construction_matrix_reports = []
            for matrix_index, item in enumerate(parsed["construction_matrices"]):
                matrix = item["matrix"]
                variants = matrix_variants(matrix)
                matrix_dir = output_dir / f"matrix_to_scan_data_{matrix_index:02d}"
                matrix_dir.mkdir(parents=True, exist_ok=True)
                variant_results = {}
                for variant_name, transform in variants:
                    transformed_margin = apply_transform(margins, transform)
                    np.save(matrix_dir / f"margin_{variant_name}.npy", transformed_margin)
                    try:
                        variant_results[variant_name] = surface_distance_report(
                            die_mesh, transformed_margin, args.units_per_mm
                        )
                    except Exception as exc:
                        variant_results[variant_name] = {"error": str(exc)}
                distance_reports.append({
                    "matrix_source": f"MatrixToScanDataFiles_{item['scope']}_{matrix_index}",
                    "assumption": "اختبار margin بعد التحويل مقابل modeldie الخام؛ إطار الإحداثيات لم يُحسم",
                    "variants": variant_results,
                })
                construction_matrix_reports.append({
                    "index": matrix_index,
                    "scope": item["scope"],
                    "row_interpretation_affine_check": bool(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)),
                    "transpose_interpretation_affine_check": bool(np.allclose(matrix.T[3], [0, 0, 0, 1], atol=1e-5)),
                })

            report = {
                "case_check": "PASS: root ConstuctionInfo; ReconstructionType=AnatomicCrown; Number=46",
                "scanfile_entries": len(scans),
                "scanfile_transformations_found": sum(bool(x["record"]["matrix_present"]) for x in scans),
                "matrix_to_scan_data_files_count": len(parsed["construction_matrices"]),
                "margin_vec3_count": int(len(margins)),
                "units_per_mm_assumption": args.units_per_mm,
                "distance_threshold_mm": 0.5,
                "scanfile_transform_reports": scan_reports,
                "matrix_to_scan_data_files_reports": construction_matrix_reports,
                "margin_to_modeldie_distance_diagnostics": distance_reports,
                "warning": (
                    "لا تعتبر أقل من 0.5 mm ضمانًا للمحاذاة. تأكد بصريًا من الاتجاه والوحدات "
                    "وموضع margin فوق PLY/modeldie في الإحداثيات نفسها. MatrixToScanDataFiles "
                    "لم تُطبّق على PLY تلقائيًا لأنها منفصلة في XML عن TransformationMatrix داخل ScanFile."
                ),
            }
            (output_dir / "diagnostic_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"\nحُفظت النتائج محليًا في: {output_dir}")
            return 0
    except Exception as exc:
        print(f"فشل الفحص: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
