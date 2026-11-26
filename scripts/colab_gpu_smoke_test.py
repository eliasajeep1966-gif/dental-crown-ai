#!/usr/bin/env python3
"""Run the one-case MADCrowner GPU smoke test inside the pinned Colab Conda env."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_batch(data_root: Path):
    from pytorch3d.ops import sample_farthest_points

    # Verify the compiled PyTorch3D CUDA FPS extension before loading the case.
    probe = torch.rand(1, 32, 3, device="cuda")
    sampled, indices = sample_farthest_points(probe, K=8)
    assert sampled.shape == (1, 8, 3) and indices.shape == (1, 8)
    print("Native CUDA FPS probe passed:", tuple(sampled.shape), flush=True)

    repo_root = Path(__file__).resolve().parents[1]
    mad_root = repo_root / "third_party" / "MADCrowner"
    sys.path.insert(0, str(mad_root))
    from mydataset.DentalDataset import IOS_Dataset

    dataset = IOS_Dataset(data_root, is_train=True, sample_points=64)
    assert len(dataset) == 1, f"Expected one case, got {len(dataset)}"
    batch = next(
        iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    )
    inputs, targets, crown_center, template, psr_grid, curvatures, margin, case_path = batch
    assert inputs.shape == (1, 128, 4), tuple(inputs.shape)
    assert targets.shape == (1, 256, 3), tuple(targets.shape)
    assert template.shape == (1, 64, 3), tuple(template.shape)
    assert psr_grid.shape == (1, 128, 128, 128), tuple(psr_grid.shape)
    assert torch.isfinite(inputs).all() and torch.isfinite(targets).all()
    assert torch.isfinite(curvatures).all() and torch.isfinite(margin).all()
    print(
        "IOS_Dataset/FPS OK:",
        {
            "inputs": tuple(inputs.shape),
            "crown": tuple(targets.shape),
            "template": tuple(template.shape),
            "psr": tuple(psr_grid.shape),
            "positive_sampled_labels": int(inputs[..., 3].sum().item()),
        },
        flush=True,
    )
    return inputs, targets, template, psr_grid, curvatures, margin


def run_training(data_root: Path, checkpoint_path: Path) -> None:
    import torch.nn.functional as F
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops import sample_farthest_points

    repo_root = Path(__file__).resolve().parents[1]
    mad_root = repo_root / "third_party" / "MADCrowner"
    sys.path.insert(0, str(mad_root))
    from models.crowndeformer import CrownDeformer
    from models.loss import curvature_penalty_loss

    inputs, targets, template, psr_grid, curvatures, margin = load_batch(data_root)
    device = torch.device("cuda")
    model = CrownDeformer(psr_resolution=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    inputs = inputs.to(device)
    targets = targets.to(device)
    template = template.to(device)
    psr_grid = psr_grid.to(device)
    curvatures = curvatures.to(device)
    margin = margin.to(device)
    target_psr = F.interpolate(
        psr_grid.unsqueeze(1), size=(32, 32, 32), mode="trilinear", align_corners=False
    ).squeeze(1)
    target_seed = sample_farthest_points(targets.contiguous(), K=64)[0]
    first_param = next(p for p in model.parameters() if p.requires_grad)
    before = first_param.detach().clone()
    loss_history = []
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for step in range(1, 11):
        optimizer.zero_grad(set_to_none=True)
        pred_psr, seed, coarse_pc, dense_cloud = model(inputs, template)
        loss_psr = F.mse_loss(pred_psr, target_psr)
        loss_margin_curvature = curvature_penalty_loss(
            dense_cloud, targets, curvatures, margin, weight_lambda=1.0
        )
        loss_seed = chamfer_distance(seed, target_seed)[0]
        loss_coarse = chamfer_distance(coarse_pc, targets)[0]
        loss = loss_psr + loss_margin_curvature + loss_seed + loss_coarse
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"step {step:02d}/10 | loss={loss_history[-1]:.6f} "
            f"| peak GPU memory={peak_gib:.2f} GiB",
            flush=True,
        )

    changed = not torch.equal(before, first_param.detach())
    assert changed, "Optimizer did not update the checked trainable parameter."
    print("Ten optimizer steps completed; parameter update confirmed.", flush=True)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": 10,
            "loss_history": loss_history,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "note": "smoke test only; random initialization; not for clinical use",
        },
        checkpoint_path,
    )
    check = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert check["steps"] == 10 and len(check["loss_history"]) == 10
    print("Checkpoint saved and reloaded:", checkpoint_path, flush=True)
    print("Final loss:", check["loss_history"][-1], flush=True)
    print("Checkpoint size (MiB):", round(checkpoint_path.stat().st_size / 1024**2, 2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dataset", "train"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the isolated Python 3.11 environment.")
    print(
        "Conda Python:", sys.version.split()[0],
        "| PyTorch:", torch.__version__,
        "| CUDA:", torch.version.cuda,
        "| GPU:", torch.cuda.get_device_name(0),
        flush=True,
    )
    if args.phase == "dataset":
        load_batch(args.data_root)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required for --phase train")
    run_training(args.data_root, args.checkpoint)


if __name__ == "__main__":
    main()
