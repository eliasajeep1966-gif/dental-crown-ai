"""CPU smoke tests for the vendored MADCrowner data loader.

PyTorch3D's FPS op is stubbed with deterministic index sampling because the
full CUDA extension is not required to test file wiring and coordinate scaling.
"""
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

import h5py
import numpy as np
import trimesh

try:
    import torch
except ImportError:  # Keep the lightweight preprocessing test suite usable without torch.
    torch = None

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "third_party" / "MADCrowner" / "mydataset" / "DentalDataset.py"


def load_dataset_module():
    if torch is None:
        raise unittest.SkipTest("PyTorch is needed for MADCrowner Dataset CPU smoke tests")

    pytorch3d = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")

    def deterministic_fps(points, K):
        if K > points.shape[1]:
            raise ValueError(f"K={K} exceeds available points={points.shape[1]}")
        indices_1d = torch.arange(K, device=points.device)
        indices = indices_1d.unsqueeze(0).expand(points.shape[0], -1)
        expanded = indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        return torch.gather(points, 1, expanded), indices

    ops.sample_farthest_points = deterministic_fps
    pytorch3d.ops = ops
    sys.modules["pytorch3d"] = pytorch3d
    sys.modules["pytorch3d.ops"] = ops

    spec = importlib.util.spec_from_file_location("madcrowner_dataset_under_test", DATASET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DentalDatasetCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_module = load_dataset_module()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.case_dir = self.root / "46" / "train" / "case_0001"
        self.case_dir.mkdir(parents=True)
        self.template_dir = self.root / "templates"
        self.template_dir.mkdir()

        crown = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
        pna = crown.copy()
        pna.apply_translation([1.0, 0.0, 0.0])
        crown.export(self.case_dir / "crown.ply")
        pna.export(self.case_dir / "pna_crop.ply")
        crown.export(self.template_dir / "46crown.ply")

        labels = np.zeros(len(pna.vertices), dtype=np.uint8)
        labels[np.argsort(np.abs(pna.vertices[:, 2]))[:12]] = 1
        self.labels = labels
        np.save(self.case_dir / "boundary_labels.npy", labels)

        with h5py.File(self.case_dir / "crown_attributes.h5", "w") as attrs:
            attrs.create_dataset("curvatures", data=np.linspace(-2.0, 2.0, len(crown.vertices), dtype=np.float32))
            attrs.create_dataset("whetherismarginline", data=np.zeros(len(crown.vertices), dtype=np.uint8))
        np.savez_compressed(self.case_dir / "psr_my.npz", psr=np.zeros((8, 8, 8), dtype=np.float16))

        self.dataset = self.dataset_module.IOS_Dataset(
            self.root, sample_points=8, template_dir=self.template_dir
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_common_center_preserves_relative_translation(self):
        torch = self.dataset_module.torch
        first = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        second = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        shared_center = torch.tensor([0.0, 0.0, 0.0])

        first_shared = self.dataset.normalize_point_cloud(first, center=shared_center)
        second_shared = self.dataset.normalize_point_cloud(second, center=shared_center)
        first_independent = self.dataset.normalize_point_cloud(first)
        second_independent = self.dataset.normalize_point_cloud(second)

        shared_midpoint_delta = second_shared.mean(dim=0)[0] - first_shared.mean(dim=0)[0]
        independent_midpoint_delta = second_independent.mean(dim=0)[0] - first_independent.mean(dim=0)[0]
        self.assertAlmostEqual(float(shared_midpoint_delta), 0.2, places=6)
        self.assertAlmostEqual(float(independent_midpoint_delta), 0.0, places=6)

    def test_curvature_minmax_normalization_and_constant_guard(self):
        source = self.case_dir / "curvature_test.h5"
        with h5py.File(source, "w") as attrs:
            attrs.create_dataset("curvatures", data=np.array([-2.0, 0.0, 2.0], dtype=np.float32))
            attrs.create_dataset("whetherismarginline", data=np.array([0, 1, 0], dtype=np.uint8))
        curvature, margin = self.dataset.get_curvature(source)
        np.testing.assert_allclose(curvature.numpy(), [0.0, 0.5, 1.0], atol=1e-6)
        np.testing.assert_allclose(margin.numpy(), [0.0, 1.0, 0.0])

        with h5py.File(source, "w") as attrs:
            attrs.create_dataset("curvatures", data=np.array([3.0, 3.0, 3.0], dtype=np.float32))
            attrs.create_dataset("whetherismarginline", data=np.array([0, 0, 1], dtype=np.uint8))
        constant, _ = self.dataset.get_curvature(source)
        np.testing.assert_allclose(constant.numpy(), [0.0, 0.0, 0.0])

    def test_template_path_is_relative_or_configurable_and_tooth_id_is_numeric(self):
        self.assertTrue(self.dataset.template_dir.is_absolute())
        template = self.dataset.get_template(46)
        self.assertEqual(tuple(template.shape), (8, 3))
        self.assertEqual(len(self.dataset), 1)
        item = self.dataset[0]
        self.assertEqual(item[-1], str(self.case_dir))

        bundled = self.dataset_module.IOS_Dataset(self.root, sample_points=8)
        self.assertTrue((bundled.template_dir / "46crown.ply").is_file())
        self.assertEqual(tuple(bundled.get_template(46).shape), (8, 3))

    def test_dataloader_reads_one_case_and_returns_consistent_tensor_shapes(self):
        pna_sampled, crown_sampled, crown_center, template, psr, curvature, margin, _ = self.dataset[0]
        self.assertEqual(tuple(pna_sampled.shape), (16, 4))
        self.assertEqual(tuple(crown_sampled.shape), (32, 3))
        self.assertEqual(tuple(template.shape), (8, 3))
        self.assertEqual(tuple(psr.shape), (8, 8, 8))
        self.assertEqual(tuple(curvature.shape), (32,))
        self.assertEqual(tuple(margin.shape), (32,))
        self.assertTrue(np.isfinite(crown_center).all())
        self.assertTrue(torch.isfinite(pna_sampled).all())
        self.assertTrue(torch.isfinite(crown_sampled).all())
        expected_shift = torch.tensor([0.1, 0.0, 0.0]).expand(16, 3)
        torch.testing.assert_close(pna_sampled[:, :3] - crown_sampled[:16], expected_shift)
        torch.testing.assert_close(
            pna_sampled[:, 3], torch.as_tensor(self.labels[:16], dtype=torch.float32)
        )


if __name__ == "__main__":
    unittest.main()
