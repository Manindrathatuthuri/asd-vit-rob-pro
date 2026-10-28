#!/usr/bin/env python
"""
predict_asd.py - run a trained CNN->ViT model on new face images.

Pipeline per image:  image -> SpatialCNNExtractor -> feature map -> tokens -> ViT -> P(ASD)

Needs cnn_feature_extractor.py, vit_asd_classifier.py, train_asd.py next to this file, plus the
best.pt written by train_asd.py.

    # one image
    python predict_asd.py --checkpoint runs/<run>/best.pt --input face.jpg
    # a folder (searched recursively), results saved to a CSV
    python predict_asd.py --checkpoint runs/<run>/best.pt --input new_faces/ --out_csv preds.csv

Output columns: path, p_asd, pred_at_threshold, pred_at_0.5, status.
  p_asd               model probability for the Autistic class (index 1).
  pred_at_threshold   1 = Autistic, 0 = Non-Autistic, using the operating threshold chosen on the
                      VALIDATION set during training (stored in the checkpoint). Override with --threshold.
  pred_at_0.5         the same with a plain 0.5 cut-off, for comparison.

This is a research screening score, not a diagnosis. The probability is only as good as the
training data: it is NOT calibrated, and it says nothing about children unlike your training set.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    pass
from train_asd import IMG_EXTS, build_transforms, load_model_from_checkpoint  # noqa: E402
from vit_asd_classifier import AUTISTIC, CLASS_NAMES  # noqa: E402


def collect_images(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        return sorted(p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
                      and not any(part.startswith(".") for part in p.relative_to(inp).parts))
    raise FileNotFoundError(f"--input '{inp}' is neither a file nor a folder")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Predict ASD vs non-ASD with a trained CNN+ViT checkpoint")
    ap.add_argument("--checkpoint", required=True, help="best.pt (or last.pt) from train_asd.py")
    ap.add_argument("--input", required=True, help="an image file or a folder of images")
    ap.add_argument("--out_csv", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="P(ASD) cut-off; default = the validation-chosen threshold stored in the checkpoint")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args(argv)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    model, ck = load_model_from_checkpoint(args.checkpoint, device)
    thr = float(ck["val_threshold"]) if args.threshold is None else args.threshold
    cfg = model.cfg
    _, tf = build_transforms(cfg.img_size[0], cfg.cnn.in_channels, 0.0)  # eval transform, same as training
    print(f"Model: epoch {ck['epoch']} | input {cfg.img_size[0]}px, {cfg.cnn.in_channels} channel(s) | "
          f"threshold {thr:.4f} ({'from checkpoint' if args.threshold is None else 'user-set'}) | device {device}")

    paths = collect_images(Path(args.input))
    if not paths:
        raise SystemExit(f"No images ({', '.join(sorted(IMG_EXTS))}) found under {args.input}")

    rows, batch, batch_paths = [], [], []

    @torch.no_grad()
    def flush() -> None:
        if not batch:
            return
        probs = F.softmax(model(torch.stack(batch).to(device)).float(), dim=-1)[:, AUTISTIC].cpu().numpy()
        for p, pr in zip(batch_paths, probs):
            rows.append([str(p), f"{pr:.6f}", int(pr >= thr), int(pr >= 0.5), "ok"])
        batch.clear()
        batch_paths.clear()

    for p in paths:
        try:
            with Image.open(p) as im:
                x = tf(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            rows.append([str(p), "", "", "", f"unreadable: {type(e).__name__}"])
            continue
        batch.append(x)
        batch_paths.append(p)
        if len(batch) == args.batch_size:
            flush()
    flush()

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "p_asd", "pred_at_threshold", "pred_at_0.5", "status"])
        w.writerows(rows)

    ok = [r for r in rows if r[4] == "ok"]
    if len(rows) <= 10:
        for r in rows:
            label = CLASS_NAMES[int(r[2])] if r[4] == "ok" else r[4]
            print(f"{r[0]}: P(ASD) = {r[1] or 'n/a'} -> {label}")
    if ok:
        n_pos = sum(int(r[2]) for r in ok)
        print(f"{len(ok)} image(s) scored, {n_pos} above threshold as {CLASS_NAMES[AUTISTIC]}, "
              f"{len(rows) - len(ok)} unreadable. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
