#!/usr/bin/env python3
"""
Fine-tune a YOLO detector to recognise food ingredients.

This is the training half of infrastructure/cnn/yolo_detector.py. It is NOT
imported by the app and is not installed into the API image — it runs on a
machine with a GPU and produces a best.pt that the app then loads.

Usage:
    # 1. put a dataset in YOLO format under datasets/ingredients/ (see README)
    # 2. train
    uv run python training/train_yolo.py --data training/ingredients.yaml

    # resume / longer run / bigger model
    uv run python training/train_yolo.py --model yolo11s.pt --epochs 200

    # then point the app at the result
    export DETECTOR_BACKEND=yolo
    export YOLO_WEIGHTS=runs/detect/ingredients/weights/best.pt

Torch here should be the CUDA build; the project pins CPU wheels for serving.
Install the GPU build into this environment only:
    uv pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO for food ingredients")
    p.add_argument(
        "--data",
        default=str(REPO_ROOT / "training" / "ingredients.yaml"),
        help="dataset YAML (paths + class names)",
    )
    p.add_argument(
        "--model",
        default="yolo11n.pt",
        help="starting checkpoint; n/s/m/l/x trade speed for accuracy",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="-1 auto-sizes to your GPU")
    p.add_argument(
        "--device",
        default=None,
        help="'0' for first GPU, 'cpu' to force CPU. Default: auto-detect.",
    )
    p.add_argument("--name", default="ingredients", help="run name under runs/detect/")
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="stop after N epochs without improvement",
    )
    return p.parse_args(argv)


def resolve_device(requested: str | None) -> str:
    """Pick a device, and say plainly when training will be slow."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    print(
        "WARNING: no CUDA device found — training on CPU will take hours to days.\n"
        "         The project pins CPU-only torch for serving; install the CUDA\n"
        "         build in this environment to train. See the module docstring.",
        file=sys.stderr,
    )
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset config not found: {data_path}", file=sys.stderr)
        print("See training/README.md for the expected layout.", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: uv sync", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    print(f"Training {args.model} on {data_path} (device={device}, epochs={args.epochs})")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        patience=args.patience,
        plots=True,
    )

    metrics = model.val()
    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\nmAP50-95: {metrics.box.map:.4f}   mAP50: {metrics.box.map50:.4f}")
    print(f"Best weights: {best}")
    print(f"\nTo serve this model:\n  export DETECTOR_BACKEND=yolo\n  export YOLO_WEIGHTS={best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
