import importlib.util
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import h5py
import numpy as np
import trimesh

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_training_case.py"
SPEC = importlib.util.spec_from_file_location("prepare_training_case", SCRIPT_PATH)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREP)


class PrepareTrainingCaseTests(unittest.TestCase):
    def test_welded_stl_is_watertight_and_psr_is_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            stl_path = pathlib.Path(temp) / "synthetic_closed.stl"
            trimesh.creation.icosphere(subdivisions=2, radius=2.0).export(stl_path)
            raw = trimesh.load(stl_path, process=False)
            welded = PREP.weld_mesh(raw)
            self.assertTrue(welded.is_watertight)
            self.assertLess(len(welded.vertices), len(raw.vertices))
            points01 = (welded.vertices / 4.0) + 0.5
            psr, scale_mode = PREP.dpsr_numpy(points01, welded.vertex_normals, resolution=32)
            self.assertEqual(psr.shape, (32, 32, 32))
            self.assertTrue(np.isfinite(psr).all())
            self.assertIn(scale_mode, {"SAP_origin_value", "p99_abs_fallback_origin_was_zero"})

    def test_full_synthetic_case_writes_five_files_and_consistent_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "synthetic_case"
            root.mkdir()
            out = pathlib.Path(temp) / "prepared_sample"

            crown = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
            crown.export(root / "46-crown_cad.stl")
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(root / "46-modeldie.stl")

            xs = np.linspace(-4.0, 4.0, 41)
            ys = np.linspace(-4.0, 4.0, 41)
            jaw_vertices = np.asarray([[x, y, -1.5] for y in ys for x in xs], dtype=np.float64)
            jaw_faces = []
            for row in range(len(ys) - 1):
                for col in range(len(xs) - 1):
                    a = row * len(xs) + col
                    b, c, d = a + 1, a + len(xs), a + len(xs) + 1
                    jaw_faces.extend([[a, b, d], [a, d, c]])
            jaw = trimesh.Trimesh(jaw_vertices, np.asarray(jaw_faces), process=False)
            jaw.export(root / "mandibular.ply")

            angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            ring_radius = np.sqrt(2.0**2 - 1.5**2)
            margin = np.column_stack((
                ring_radius * np.cos(angles),
                ring_radius * np.sin(angles),
                np.full_like(angles, -1.5),
            ))
            construction = ET.Element("ConstuctionInfo")
            ET.SubElement(construction, "ReconstructionType").text = "AnatomicCrown"
            ET.SubElement(construction, "Number").text = "46"
            margin_el = ET.SubElement(construction, "Margin")
            for x, y, z in margin:
                vec = ET.SubElement(margin_el, "Vec3")
                ET.SubElement(vec, "x").text = str(x)
                ET.SubElement(vec, "y").text = str(y)
                ET.SubElement(vec, "z").text = str(z)
            scan_files = ET.SubElement(construction, "ScanFiles")
            scan_file = ET.SubElement(scan_files, "ScanFile")
            ET.SubElement(scan_file, "FileName").text = "mandibular.ply"
            matrix = ET.SubElement(scan_file, "TransformationMatrix")
            for row in range(4):
                for col in range(4):
                    ET.SubElement(matrix, f"_{row}{col}").text = str(float(row == col))
            ET.ElementTree(construction).write(
                root / "synthetic.constructionInfo", encoding="utf-8", xml_declaration=True
            )

            args = SimpleNamespace(
                case=str(root), out=str(out), constructioninfo=None, crown=None,
                modeldie=None, jaw_hint="mandibular", units_per_mm=1.0,
                crop_size_mm=20.0, margin_tolerance_mm=0.15, max_snap_mm=1.5,
                max_crown_snap_mm=0.75, max_jaw_margin_mm=1.5,
                max_crown_margin_mm=0.1, psr_resolution=32,
            )
            report = PREP.build_sample(args)

            expected_files = {
                "crown.ply", "pna_crop.ply", "boundary_labels.npy",
                "crown_attributes.h5", "psr_my.npz", "prep_report.json",
            }
            self.assertEqual({path.name for path in out.iterdir()}, expected_files)
            crown_saved = trimesh.load(out / "crown.ply", process=False)
            pna_saved = trimesh.load(out / "pna_crop.ply", process=False)
            labels = np.load(out / "boundary_labels.npy", allow_pickle=False)
            self.assertEqual(len(pna_saved.vertices), len(labels))
            self.assertGreater(np.count_nonzero(labels), 0)
            with h5py.File(out / "crown_attributes.h5", "r") as attrs:
                self.assertEqual(set(attrs.keys()), {"curvatures", "whetherismarginline"})
                self.assertEqual(attrs["curvatures"].shape, (len(crown_saved.vertices),))
                self.assertEqual(attrs["whetherismarginline"].shape, (len(crown_saved.vertices),))
            with np.load(out / "psr_my.npz", allow_pickle=False) as psr_file:
                self.assertEqual(psr_file["psr"].shape, (32, 32, 32))
                self.assertTrue(np.isfinite(psr_file["psr"]).all())
            self.assertFalse(report["training_readiness"]["approved_for_training"])


if __name__ == "__main__":
    unittest.main()
