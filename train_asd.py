#!/usr/bin/env python
"""
train_asd.py - end-to-end training of CNNVITASDClassifier for ASD facial-image screening.

Files needed side by side: cnn_feature_extractor.py, vit_asd_classifier.py, train_asd.py.
Dependencies: torch, torchvision, numpy, scikit-learn, Pillow, tqdm (optional).

Run
---
    python train_asd.py --data_root /path/to/dataset_root

    dataset_root/
        Autistic/        (also: autism, ASD, ...)
            img001.jpg
        Non_Autistic/    (also: TD, control, typical, healthy, ...)
            img001.jpg

Class folders are matched case-insensitively and mapped to
    index 0 = Non-Autistic, index 1 = Autistic   (P(ASD) = probs[:, 1]).
If auto-detection guesses wrong or refuses, pass --asd_dir NAME --control_dir NAME.
A pre-split layout (root/train/<classes>, root/val/<classes>, root/test/<classes>) is
pooled and re-split by this script, because splits made at image level leak identity.

Colab
-----
    from google.colab import drive
    drive.mount('/content/drive')
    # Runtime > Change runtime type > GPU. Upload the three .py files to /content (or Drive), then:
    !python /content/train_asd.py --data_root "/content/drive/MyDrive/asd_faces" \\
            --out_dir "/content/drive/MyDrive/asd_runs"
    # Writing --out_dir to Drive keeps checkpoints if the Colab runtime is recycled.
    # There is no resume-from-checkpoint: a recycled runtime means restarting that run.

What it does
------------
1. Scans images (decodes each once: unreadable files are dropped; MD5 of each file is kept).
2. Splits by SUBJECT when it can find subject IDs, otherwise by image, and ALWAYS keeps
   byte-identical duplicate files in the same split. Prints images / subjects per split.
3. Trains with AdamW + warmup + cosine, grad clipping, optional AMP, label smoothing,
   optional class weighting; early stops on val AUROC (or balanced accuracy).
4. Keeps best.pt (by monitored val metric) and last.pt; logs every epoch.
5. After training: loads best.pt, picks an operating threshold on VAL ONLY, and reports the
   held-out TEST set once, at threshold 0.5 and at the val-chosen threshold, with a
   subject-level bootstrap CI on AUROC. Writes run_log.json and per-image predictions CSVs.

Subject-ID heuristics (verify them on your data; they are guesses from filenames)
--------------------------------------------------------------------------------
Tried in order, first reliable one wins. A scheme is reliable only if >=90% of files match,
IDs repeat (images/subject >= 1.1; otherwise the "ID" is just an image index such as the
001 in img001.jpg) and every class has >= 10 distinct subjects:
    --subject_regex (your regex, group 1 = ID)  >  class/<subject_folder>/img.jpg
    >  keyword + number ("subject_12", "child03", "patient-7", "id5")
    >  leading number pair ("012_3.jpg" -> subject 12)
The class is part of the subject key (subject 3 of one class is not subject 3 of the other).

Recommended starting settings (conventional starting points for ~1000-3000 images; NOT
tuned on any ASD dataset, so treat as the first row of a sweep)
--------------------------------------------------------------------------------
  default           --depth 4 --embed_dim 256 --lr 3e-4 --weight_decay 0.05 --batch_size 32
                    --drop_rate 0.2 --drop_path_rate 0.1 --label_smoothing 0.1 --aug_strength 1.0
  ~1000 images      --depth 2 --drop_rate 0.3 --aug_strength 1.5 --img_size 160 --batch_size 16
  ~3000 images      --depth 4 --drop_rate 0.2 --aug_strength 1.0
  small GPU memory  --batch_size 16 --img_size 160   (keep --downsample_factor 8; factors 2/4 need --patch_size >= 2)
  fully reproducible --deterministic   (slower; also forces the plain attention kernel)
  screening setting --sens_target 0.90 (threshold on val: highest specificity at >= 90% sens.)
Expect a from-scratch 5-6M-parameter model on a few thousand faces to overfit: watch the gap
between train and val AUROC, and compare against a pretrained backbone before trusting it.

Example console output
----------------------
(Real output from 241 synthetic noise images, tiny model, 3 epochs, CPU. The numbers are meaningless;
    it only shows the format. The AUROC CI and checkpoint-path lines that follow the confusion matrices are omitted.)

    Run dir: runs/example
    Device: cpu
    Class mapping: ['Non_Autistic'] -> 0 (non_autistic) | ['Autistic'] -> 1 (autistic)
    Images: 241 (ASD 127, non-ASD 114); shorter side min/median/max = 72/72/72 px
    Subject IDs: scheme 'keyword+number in filename' | 36 ASD + 30 non-ASD subjects | 3.65 images/subject | 100% of files matched. These are filename heuristics: spot-check a few before trusting them.
    Found 6 byte-identical duplicate file(s); each duplicate set is kept inside one split. Near-duplicates (resized / re-encoded copies) are NOT detected.

    Split statistics
    split   images  ASD img non-ASD img  ASD subjects  non-ASD subjects
    train      193      102          91            25                24
    val         24       12          12             3                 3
    test        24       13          11             2                 3
    Train class counts [non-ASD, ASD] = [91, 102] (minority/majority 0.89) -> imbalance handling: none

    Model: CNN 1.14M + ViT 0.12M + head 0.3K = 1.26M params | token grid 8x8 = 64 tokens | train images 193
    Mixed precision: off

    Training: up to 3 epochs, monitor = val auroc, patience 20 (after warmup). Train metrics are measured on augmented batches with dropout on, so they under-read.

    ep 001/3 lr 3.00e-04 | train loss 0.6463 acc 0.663 bacc 0.659 auroc 0.696 | val loss 0.3197 acc 1.000 bacc 1.000 auroc 1.000 sens 1.000 spec 1.000 prec 1.000 f1 1.000 | 6s *best*
    ep 002/3 lr 1.69e-04 | train loss 0.6445 acc 0.679 bacc 0.685 auroc 0.662 | val loss 0.3852 acc 1.000 bacc 1.000 auroc 1.000 sens 1.000 spec 1.000 prec 1.000 f1 1.000 | 6s
    ep 003/3 lr 4.08e-06 | train loss 0.5835 acc 0.705 bacc 0.712 auroc 0.790 | val loss 0.3311 acc 1.000 bacc 1.000 auroc 1.000 sens 1.000 spec 1.000 prec 1.000 f1 1.000 | 6s

    ==============================================================================
    SUMMARY
    Best epoch 1 (val auroc 1.0000) of 3 run | params 1.26M | time 0.4 min
    Threshold on val (Youden's J): 0.7623  (0.5 is NOT a calibrated clinical operating point)

    val at 0.5: acc 1.000 bacc 1.000 auroc 1.000 sens 1.000 spec 1.000 prec 1.000 f1 1.000
                     pred non-ASD   pred ASD
      true non-ASD           12          0
      true ASD                0         12

    val at val thr: acc 1.000 bacc 1.000 auroc 1.000 sens 1.000 spec 1.000 prec 1.000 f1 1.000 <- threshold was tuned on this set; optimistic
                     pred non-ASD   pred ASD
      true non-ASD           12          0
      true ASD                0         12

    test at 0.5: acc 1.000 bacc 1.000 auroc 1.000 sens 1.000 spec 1.000 prec 1.000 f1 1.000
                     pred non-ASD   pred ASD
      true non-ASD           11          0
      true ASD                0         13

    test at val thr: acc 1.000 bacc 1.000 auroc 1.000 sens 1.000 spec 1.000 prec 1.000 f1 1.000
                     pred non-ASD   pred ASD
      true non-ASD           11          0
      true ASD                0         13
    val AUROC 95% bootstrap CI (resampling 6 subjects): [1.000, 1.000]   <- too few units for this interval to mean much
    test AUROC 95% bootstrap CI (resampling 5 subjects): [1.000, 1.000]   <- too few units for this interval to mean much

    Best checkpoint: runs/example/best.pt
    Last checkpoint: runs/example/last.pt
    Log: runs/example/run_log.json
"""

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, get_args, get_origin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # exec'd in a notebook cell without __file__
    sys.path.insert(0, os.getcwd())
from cnn_feature_extractor import CNNExtractorConfig  # noqa: E402  (models are used unchanged)
from vit_asd_classifier import (AUTISTIC, CLASS_NAMES, NON_AUTISTIC, ASDModelConfig,  # noqa: E402
                                CNNVITASDClassifier, ViTConfig)

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_DIR_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing", "dev"}
NAN = float("nan")


# --------------------------------------------------------------------------- #
# Configuration (every field becomes a --flag)
# --------------------------------------------------------------------------- #
def _f(default, help, choices=None):
    return field(default=default, metadata={"help": help, "choices": choices})


@dataclass
class TrainConfig:
    # data
    data_root: Optional[str] = _f(None, "folder containing the two class folders")
    asd_dir: Optional[str] = _f(None, "name of the Autistic class folder (override auto-detection)")
    control_dir: Optional[str] = _f(None, "name of the Non-Autistic class folder (override auto-detection)")
    subject_regex: Optional[str] = _f(None, "regex whose group 1 is the subject ID, applied to the filename stem")
    val_frac: float = _f(0.1, "validation fraction of the whole dataset")
    test_frac: float = _f(0.1, "test fraction of the whole dataset (0 = no test set, train/val only)")
    # images / augmentation
    img_size: int = _f(224, "input resolution (square). 224 with downsample 8 gives 784 tokens")
    channels: int = _f(3, "1 = grayscale model, 3 = RGB model", choices=[1, 3])
    aug_strength: float = _f(1.0, "0 = flip only; 1 = default; 2 = strong (scales crop, rotation, jitter, erasing)")
    # model
    downsample_factor: int = _f(8, "CNN total stride: 2, 4 or 8 (2 and 4 need --patch_size >= 2 at 224px)", choices=[2, 4, 8])
    patch_size: int = _f(1, "CNN cells merged per token side in the ViT")
    embed_dim: int = _f(256, "ViT width")
    depth: int = _f(4, "ViT blocks (4 is a deliberately small default for small data)")
    num_heads: int = _f(4, "attention heads, must divide embed_dim")
    mlp_ratio: float = _f(4.0, "MLP hidden = mlp_ratio * embed_dim")
    drop_rate: float = _f(0.2, "ViT dropout")
    attn_drop_rate: float = _f(0.1, "ViT attention dropout")
    drop_path_rate: float = _f(0.1, "ViT max stochastic depth")
    # optimisation
    epochs: int = _f(100, "maximum epochs")
    batch_size: int = _f(32, "training batch size")
    lr: float = _f(3e-4, "peak learning rate")
    weight_decay: float = _f(0.05, "AdamW weight decay (not applied to norms, biases, cls token, pos-embed)")
    warmup_epochs: int = _f(5, "linear LR warmup epochs")
    min_lr_ratio: float = _f(0.01, "final LR = lr * min_lr_ratio (cosine)")
    label_smoothing: float = _f(0.1, "label smoothing for the training loss")
    grad_clip: float = _f(1.0, "max gradient norm (0 disables)")
    imbalance: str = _f("auto", "none | loss (weighted CE) | sampler (weighted sampling) | auto (loss if minority < 80 percent of majority)",
                        choices=["auto", "none", "loss", "sampler"])
    monitor: str = _f("auroc", "metric for best checkpoint / early stopping", choices=["auroc", "balanced_acc"])
    patience: int = _f(20, "stop after this many epochs without improvement (only after warmup)")
    sens_target: Optional[float] = _f(None, "if set (e.g. 0.9) choose the val threshold with highest specificity at >= this sensitivity; else Youden's J")
    bootstrap: int = _f(1000, "bootstrap resamples (by subject) for the AUROC CI; 0 disables")
    # system
    seed: int = _f(42, "seed for python / numpy / torch / splits")
    num_workers: int = _f(-1, "DataLoader workers (-1 = min(4, cpu count))")
    amp: bool = _f(True, "mixed precision on CUDA (bf16 if the GPU supports it, else fp16 + GradScaler)")
    deterministic: bool = _f(False, "deterministic algorithms and attention kernel (slower)")
    device: str = _f("auto", "auto | cpu | cuda")
    out_dir: str = _f("runs", "where run folders are created")
    run_name: Optional[str] = _f(None, "run folder name (default: timestamp_seed)")
    progress: bool = _f(True, "show tqdm progress bars")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CNN->ViT ASD facial screening model",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for f in fields(TrainConfig):
        t = f.type
        if get_origin(t) is Union:
            t = [a for a in get_args(t) if a is not type(None)][0]
        helptxt = f.metadata.get("help", "")
        if t is bool:
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default, help=helptxt)
        else:
            p.add_argument(f"--{f.name}", type=t, default=f.default, choices=f.metadata.get("choices"), help=helptxt)
    return p


class RunLogger:
    """print() + append to <run_dir>/train.log."""

    def __init__(self) -> None:
        self.path: Optional[Path] = None

    def attach(self, path: Path) -> None:
        self.path = path

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.path is not None:
            with open(self.path, "a") as fh:
                fh.write(msg + "\n")


log = RunLogger()


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


_NEG_FIRST = {"non", "no", "not", "td", "nt", "control", "controls", "typical", "typically",
              "healthy", "normal", "neurotypical", "nonautistic", "nonasd"}
_NEG_ANY = {"negative", "neg"}


def classify_folder(name: str) -> Optional[int]:
    """1 = autistic folder, 0 = non-autistic folder, None = unrecognised."""
    n = _norm(name)
    toks = n.split("_")
    if toks[0] in _NEG_FIRST or n.startswith("non") or any(t in _NEG_ANY for t in toks):
        return NON_AUTISTIC
    if "autis" in n or "asd" in toks:
        return AUTISTIC
    return None


def find_class_dirs(root: Path, asd_dir: Optional[str], control_dir: Optional[str]
                    ) -> Tuple[List[Path], List[Path], bool]:
    """Return (autistic_dirs, control_dirs, pooled). Pools train/val/test sub-roots if present."""
    if not root.is_dir():
        raise FileNotFoundError(f"data_root '{root}' does not exist or is not a directory")

    def subdirs(r: Path) -> List[Path]:
        return sorted(d for d in r.iterdir() if d.is_dir() and not d.name.startswith("."))

    def resolve(r: Path) -> Optional[Tuple[Path, Path]]:
        subs = subdirs(r)
        if asd_dir or control_dir:
            by_name = {d.name.lower(): d for d in subs}
            a = by_name.get((asd_dir or "").lower())
            c = by_name.get((control_dir or "").lower())
            return (a, c) if a and c else None
        pos = [d for d in subs if classify_folder(d.name) == AUTISTIC]
        neg = [d for d in subs if classify_folder(d.name) == NON_AUTISTIC]
        return (pos[0], neg[0]) if len(pos) == 1 and len(neg) == 1 else None

    res = resolve(root)
    if res is not None:
        return [res[0]], [res[1]], False
    subs = subdirs(root)
    if subs and all(_norm(d.name) in SPLIT_DIR_NAMES for d in subs):
        pairs = [resolve(d) for d in subs]
        if all(p is not None for p in pairs):
            return [p[0] for p in pairs], [p[1] for p in pairs], True
    raise ValueError(
        f"Could not identify exactly one Autistic and one Non-Autistic folder under '{root}'.\n"
        f"  Sub-folders found: {[d.name for d in subs] or 'none'}\n"
        f"  Expected e.g. Autistic/ + Non_Autistic/ (or ASD/ + TD/, autism/ + control/).\n"
        f"  Fixes: point --data_root at the folder that directly contains the two class folders, "
        f"or name them with --asd_dir NAME --control_dir NAME.")


@dataclass
class Sample:
    path: str
    label: int
    md5: str
    nested: Optional[str]  # first sub-folder below the class folder, if any
    size: Tuple[int, int]


def scan_images(asd_dirs: Sequence[Path], ctl_dirs: Sequence[Path]) -> Tuple[List[Sample], List[str]]:
    """Decode every image once (drops unreadable files) and hash its bytes."""
    samples: List[Sample] = []
    bad: List[str] = []
    for label, dirs in ((AUTISTIC, asd_dirs), (NON_AUTISTIC, ctl_dirs)):
        for d in dirs:
            for p in sorted(d.rglob("*")):
                rel = p.relative_to(d)
                if (not p.is_file() or p.suffix.lower() not in IMG_EXTS
                        or any(part.startswith(".") for part in rel.parts)):
                    continue
                try:
                    data = p.read_bytes()
                    with Image.open(io.BytesIO(data)) as im:
                        im.load()  # full decode: catches truncated files before they kill training
                        size = im.size
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{p} ({type(e).__name__})")
                    continue
                nested = rel.parts[0] if len(rel.parts) > 1 else None
                samples.append(Sample(str(p), label, hashlib.md5(data).hexdigest(), nested, size))
    return samples, bad


# --------------------------------------------------------------------------- #
# Subject inference + grouping
# --------------------------------------------------------------------------- #
_KEYWORD_RE = re.compile(
    r"(?<![a-z])(?:subject|subj|sub|child|kid|patient|person|participant|pid|id)[_\-\s.]?0*(\d+)", re.I)
_LEADING_PAIR_RE = re.compile(r"^0*(\d+)[_\-\s.]+\d+$")


def _norm_id(x: str) -> str:
    return x.lstrip("0") or "0" if x.isdigit() else x


def infer_subjects(samples: List[Sample], user_regex: Optional[str]
                   ) -> Tuple[Optional[List[Optional[str]]], str, Dict[str, float]]:
    """Return (per-sample subject ID or None, scheme name, stats). IDs are None when unmatched."""
    n = len(samples)
    stems = [Path(s.path).stem for s in samples]

    def by_regex(rx: re.Pattern) -> List[Optional[str]]:
        out = []
        for st in stems:
            m = rx.search(st)
            out.append(_norm_id(m.group(1)) if m else None)
        return out

    candidates: List[Tuple[str, List[Optional[str]]]] = []
    if user_regex:
        candidates.append(("custom --subject_regex", by_regex(re.compile(user_regex))))
    candidates.append(("class sub-folders", [s.nested for s in samples]))
    candidates.append(("keyword+number in filename", by_regex(_KEYWORD_RE)))
    candidates.append(("leading number pair in filename", by_regex(_LEADING_PAIR_RE)))

    best_stats: Dict[str, float] = {}
    for name, ids in candidates:
        matched = [i for i, v in enumerate(ids) if v is not None]
        cov = len(matched) / max(n, 1)
        keys = {(samples[i].label, ids[i]) for i in matched}
        per_class = Counter(k[0] for k in keys)
        ratio = len(matched) / max(len(keys), 1)
        stats = {"coverage": cov, "images_per_subject": ratio,
                 "subjects_autistic": per_class.get(AUTISTIC, 0), "subjects_non_autistic": per_class.get(NON_AUTISTIC, 0)}
        ok = (cov >= 0.9 and ratio >= 1.1 and per_class.get(AUTISTIC, 0) >= 10 and per_class.get(NON_AUTISTIC, 0) >= 10)
        if ok:
            return ids, name, stats
        best_stats = best_stats or stats
    return None, "none", best_stats


def build_groups(samples: List[Sample], ids: Optional[List[Optional[str]]]) -> Tuple[np.ndarray, List[str], int, int]:
    """Group id per sample. Subject-level if IDs known, else per image. Byte-identical files
    are always merged into one group (union-find). Returns (groups, group_names, n_dup_files,
    n_cross_class_dup_groups)."""
    names = []
    for i, s in enumerate(samples):
        names.append(f"{s.label}:{ids[i]}" if ids is not None and ids[i] is not None else f"{s.label}:img{i}")
    parent = {nm: nm for nm in names}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first_by_hash: Dict[str, str] = {}
    n_dup = 0
    for nm, s in zip(names, samples):
        if s.md5 in first_by_hash:
            n_dup += 1
            ra, rb = find(nm), find(first_by_hash[s.md5])
            if ra != rb:
                parent[ra] = rb
        else:
            first_by_hash[s.md5] = nm
    roots = [find(nm) for nm in names]
    uniq = {r: j for j, r in enumerate(dict.fromkeys(roots))}
    groups = np.array([uniq[r] for r in roots])
    labels_by_group: Dict[int, set] = {}
    for g, s in zip(groups, samples):
        labels_by_group.setdefault(int(g), set()).add(s.label)
    cross = sum(1 for v in labels_by_group.values() if len(v) > 1)
    return groups, [r for r in uniq], n_dup, cross


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _group_stratified_holdout(idx: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                              frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out ~frac of idx; groups never straddle the cut; class balance approximately kept."""
    n_splits = max(2, int(round(1.0 / frac)))
    last_err: Optional[Exception] = None
    for attempt in range(30):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        try:
            rest, held = next(iter(sgkf.split(idx, labels[idx], groups[idx])))
        except ValueError as e:
            last_err = e
            continue
        if len(set(labels[idx[held]])) == 2 and len(set(labels[idx[rest]])) == 2:
            return idx[rest], idx[held]
    raise ValueError(f"Could not form a split with both classes on each side (frac={frac}). "
                     f"Too few subjects/images per class? ({last_err})")


def make_splits(labels: np.ndarray, groups: np.ndarray, val_frac: float, test_frac: float,
                seed: int) -> Dict[str, np.ndarray]:
    if not 0 < val_frac < 0.5 or not 0 <= test_frac < 0.5:
        raise ValueError("need 0 < val_frac < 0.5 and 0 <= test_frac < 0.5")
    idx = np.arange(len(labels))
    out: Dict[str, np.ndarray] = {}
    rest = idx
    if test_frac > 0:
        rest, out["test"] = _group_stratified_holdout(idx, labels, groups, test_frac, seed)
    train, val = _group_stratified_holdout(rest, labels, groups, val_frac / (1 - test_frac), seed + 1000)
    out["train"], out["val"] = train, val
    # hard guarantee: no group appears in two splits
    names = list(out)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = set(groups[out[names[i]]]) & set(groups[out[names[j]]])
            assert not shared, f"group leakage between {names[i]} and {names[j]}: {len(shared)} groups"
    return {k: out[k] for k in ("train", "val", "test") if k in out}


def describe_splits(splits: Dict[str, np.ndarray], labels: np.ndarray, groups: np.ndarray, subject_level: bool) -> Dict:
    log("\nSplit statistics" + ("" if subject_level else " (image-level: 'groups' = images / exact-duplicate clusters)"))
    unit = "subjects" if subject_level else "groups"
    log(f"{'split':<6}{'images':>8}{'ASD img':>9}{'non-ASD img':>12}{'ASD ' + unit:>14}{'non-ASD ' + unit:>18}")
    stats = {}
    for name, ix in splits.items():
        lab, grp = labels[ix], groups[ix]
        n1, n0 = int((lab == AUTISTIC).sum()), int((lab == NON_AUTISTIC).sum())
        g1, g0 = len(set(grp[lab == AUTISTIC])), len(set(grp[lab == NON_AUTISTIC]))
        log(f"{name:<6}{len(ix):>8}{n1:>9}{n0:>12}{g1:>14}{g0:>18}")
        stats[name] = {"images": len(ix), "asd_images": n1, "non_asd_images": n0,
                       f"asd_{unit}": g1, f"non_asd_{unit}": g0}
        if min(n1, n0) < 10:
            log(f"  WARNING: '{name}' has fewer than 10 images in one class; its metrics will be very noisy.")
    return stats


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int, channels: int, strength: float) -> Tuple[T.Compose, T.Compose]:
    """Train and eval transforms. Colour jitter runs on RGB, then optional grayscale."""
    norm = T.Normalize([0.5] * channels, [0.5] * channels)  # -> [-1, 1]; the model is trained from scratch
    gray = [T.Grayscale(1)] if channels == 1 else []
    evalt = T.Compose([T.Resize(img_size), T.CenterCrop(img_size), *gray, T.ToTensor(), norm])
    s = max(0.0, strength)
    if s == 0:
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.RandomHorizontalFlip(),
                          *gray, T.ToTensor(), norm]), evalt
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(max(0.25, 1 - 0.35 * s), 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10 * s, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomAffine(0, translate=(0.04 * s, 0.04 * s), scale=(1 - 0.08 * s, 1 + 0.08 * s),
                       interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=min(0.3 * s, 0.9), contrast=min(0.3 * s, 0.9),
                      saturation=min(0.2 * s, 0.9), hue=min(0.02 * s, 0.4)),
        *gray, T.ToTensor(), norm,
        T.RandomErasing(p=min(0.25 * s, 0.5), scale=(0.02, 0.08), value=0),
    ])
    return train, evalt


class FaceDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: Sequence[int], transform) -> None:
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")  # handles grayscale / RGBA / rotated phone photos
        return self.transform(im), self.labels[i]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y: np.ndarray, p1: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """All metrics with Autistic (1) as the positive class. p1 = P(ASD)."""
    pred = (p1 >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    sens = tp / (tp + fn) if (tp + fn) else NAN
    spec = tn / (tn + fp) if (tn + fp) else NAN
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)) if both else NAN,
        "auroc": float(roc_auc_score(y, p1)) if both else NAN,
        "sensitivity": float(sens), "specificity": float(spec),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def choose_threshold(y: np.ndarray, p1: np.ndarray, sens_target: Optional[float]) -> float:
    """Operating point from VAL ONLY. Youden's J, or max specificity subject to sensitivity >= target."""
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p1)
    if sens_target is None:
        k = int(np.argmax(tpr - fpr))
    else:
        ok = np.where(tpr >= sens_target)[0]
        k = int(ok[np.argmin(fpr[ok])]) if len(ok) else int(np.argmax(tpr))
    t = float(thr[k])
    return t if math.isfinite(t) else 1.0001  # roc_curve's first threshold is +inf (predict nobody)


def bootstrap_auroc(y: np.ndarray, p1: np.ndarray, groups: np.ndarray, n: int, seed: int
                    ) -> Tuple[float, float]:
    """95% CI for AUROC, resampling whole subjects/groups (images of one child are not independent)."""
    if n <= 0 or len(np.unique(y)) < 2:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[ix])) == 2:
            vals.append(roc_auc_score(y[ix], p1[ix]))
    if len(vals) < 10:
        return NAN, NAN
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt_metrics(m: Dict[str, float], keys=("accuracy", "balanced_acc", "auroc", "sensitivity", "specificity", "precision", "f1")) -> str:
    short = {"accuracy": "acc", "balanced_acc": "bacc", "auroc": "auroc", "sensitivity": "sens",
             "specificity": "spec", "precision": "prec", "f1": "f1"}
    return " ".join(f"{short[k]} {m[k]:.3f}" for k in keys)


def fmt_confusion(m: Dict[str, float]) -> str:
    return ("                 pred non-ASD   pred ASD\n"
            f"  true non-ASD   {m['tn']:>10d}   {m['fp']:>8d}\n"
            f"  true ASD       {m['fn']:>10d}   {m['tp']:>8d}")


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def make_optimizer(model: nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    """AdamW; no weight decay on norms / biases / layer-scale (1-D) or cls token / pos-embed."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "cls_token" in name or "pos_embed" in name else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999))


def lr_at(step: int, total: int, warmup: int, base: float, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def train_one_epoch(model, loader, opt, scaler, criterion, cfg: TrainConfig, device, autocast,
                    step: int, total_steps: int, warmup_steps: int, progress: bool):
    model.train()
    ys, ps, loss_sum, n_seen, skipped, consecutive = [], [], 0.0, 0, 0, 0
    it = tqdm(loader, desc="train", leave=False, disable=not progress) if tqdm else loader
    lr = cfg.lr
    for x, y in it:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        lr = lr_at(step, total_steps, warmup_steps, cfg.lr, cfg.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        step += 1
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
        loss = criterion(logits.float(), y)
        if not torch.isfinite(loss):
            skipped, consecutive = skipped + 1, consecutive + 1
            if consecutive >= 10:
                raise FloatingPointError("10 consecutive non-finite losses; lower --lr or use --no-amp")
            continue
        consecutive = 0
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        if not scaler.is_enabled() and not torch.isfinite(gn):  # fp32/bf16: the scaler won't skip for us
            skipped += 1
            continue
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * len(y)
        n_seen += len(y)
        ys.append(y.cpu().numpy())
        ps.append(F.softmax(logits.detach().float(), dim=-1)[:, AUTISTIC].cpu().numpy())
    if n_seen == 0:
        raise FloatingPointError("every batch in the epoch was skipped as non-finite")
    m = compute_metrics(np.concatenate(ys), np.concatenate(ps))
    m["loss"] = loss_sum / n_seen
    m["skipped_batches"] = skipped
    return m, step, lr


@torch.no_grad()
def evaluate(model, loader, device, autocast, progress: bool = False) -> Dict[str, np.ndarray]:
    """Unshuffled pass -> {'loss', 'p1', 'y'} in dataset order. Plain (unsmoothed, unweighted) CE."""
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    it = tqdm(loader, desc="eval", leave=False, disable=not progress) if tqdm else loader
    for x, y in it:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            logits = model(x)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(F.softmax(logits, dim=-1)[:, AUTISTIC].cpu().numpy())
    return {"loss": loss_sum / max(n, 1), "p1": np.concatenate(ps), "y": np.concatenate(ys)}


def build_model(cfg: TrainConfig) -> CNNVITASDClassifier:
    mcfg = ASDModelConfig(
        cnn=CNNExtractorConfig(in_channels=cfg.channels, downsample_factor=cfg.downsample_factor),
        vit=ViTConfig(embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                      drop_rate=cfg.drop_rate, attn_drop_rate=cfg.attn_drop_rate,
                      drop_path_rate=cfg.drop_path_rate, patch_size=cfg.patch_size,
                      deterministic_attention=cfg.deterministic),
        img_size=cfg.img_size)
    return CNNVITASDClassifier(mcfg)


def load_model_from_checkpoint(path: str, device: str = "cpu") -> Tuple[CNNVITASDClassifier, dict]:
    """Rebuild the model from a checkpoint written by this script (for inference / later analysis)."""
    ck = torch.load(path, map_location=device, weights_only=True)
    d = ck["model_cfg"]
    mcfg = ASDModelConfig(cnn=CNNExtractorConfig(**d["cnn"]), vit=ViTConfig(**d["vit"]), img_size=tuple(d["img_size"]))
    model = CNNVITASDClassifier(mcfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_predictions(path: Path, ix: np.ndarray, samples: List[Sample], group_names: List[str],
                      groups: np.ndarray, p1: np.ndarray, split: str, thr: float) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "split", "subject_or_group", "label", "p_asd", "pred_at_val_threshold"])
        for k, i in enumerate(ix):
            w.writerow([samples[i].path, split, group_names[groups[i]], samples[i].label,
                        f"{p1[k]:.6f}", int(p1[k] >= thr)])


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def run(cfg: TrainConfig) -> Dict:
    t_start = time.time()
    if not cfg.data_root:
        raise SystemExit("--data_root is required (folder containing the Autistic and Non_Autistic folders)")
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise SystemExit("--epochs and --batch_size must be >= 1")
    device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) or cfg.device == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    set_seed(cfg.seed, cfg.deterministic)

    run_dir = Path(cfg.out_dir) / (cfg.run_name or f"{time.strftime('%Y%m%d-%H%M%S')}_seed{cfg.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log.attach(run_dir / "train.log")
    log(f"Run dir: {run_dir}\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---- 1. discover + scan -------------------------------------------------------------- #
    asd_dirs, ctl_dirs, pooled = find_class_dirs(Path(cfg.data_root), cfg.asd_dir, cfg.control_dir)
    log(f"Class mapping: {[d.name for d in ctl_dirs]} -> {NON_AUTISTIC} ({CLASS_NAMES[0]}) | "
        f"{[d.name for d in asd_dirs]} -> {AUTISTIC} ({CLASS_NAMES[1]})")
    if pooled:
        log("NOTE: pre-split train/val/test folders found. They are pooled and re-split here, because "
            "existing splits are usually image-level and leak identity.")
    samples, bad = scan_images(asd_dirs, ctl_dirs)
    if bad:
        log(f"WARNING: skipped {len(bad)} unreadable image file(s), e.g. {bad[:3]}")
    labels = np.array([s.label for s in samples])
    if len(samples) == 0 or len(np.unique(labels)) < 2:
        raise ValueError(f"Need images in both class folders; found {Counter(labels.tolist())} readable images.")
    sizes = np.array([min(s.size) for s in samples])
    log(f"Images: {len(samples)} (ASD {int((labels == AUTISTIC).sum())}, non-ASD {int((labels == NON_AUTISTIC).sum())}); "
        f"shorter side min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()} px")
    if np.median(sizes) < cfg.img_size / 2:
        log(f"WARNING: median image is far smaller than --img_size {cfg.img_size}; consider a smaller --img_size.")

    # ---- 2. subjects, groups, splits ------------------------------------------------------ #
    ids, scheme, st = infer_subjects(samples, cfg.subject_regex)
    subject_level = ids is not None
    groups, group_names, n_dup, n_cross = build_groups(samples, ids)
    if subject_level:
        log(f"Subject IDs: scheme '{scheme}' | {int(st['subjects_autistic'])} ASD + {int(st['subjects_non_autistic'])} "
            f"non-ASD subjects | {st['images_per_subject']:.2f} images/subject | {st['coverage']:.0%} of files matched. "
            "These are filename heuristics: spot-check a few before trusting them.")
    else:
        log("WARNING: no reliable subject IDs found (tried: " + ("custom regex, " if cfg.subject_regex else "") +
            "class sub-folders, keyword+number, leading number pair). Falling back to a STRATIFIED IMAGE-LEVEL split: "
            "photos of the same child can land in train and test, which inflates every metric. "
            "Pass --subject_regex or arrange files as <class>/<subject>/img.jpg if you can recover identities.")
    if n_dup:
        log(f"Found {n_dup} byte-identical duplicate file(s); each duplicate set is kept inside one split. "
            "Near-duplicates (resized / re-encoded copies) are NOT detected.")
    if n_cross:
        log(f"WARNING: {n_cross} group(s) contain the same image under BOTH class labels. Label noise in the dataset.")

    splits = make_splits(labels, groups, cfg.val_frac, cfg.test_frac, cfg.seed)
    split_stats = describe_splits(splits, labels, groups, subject_level)
    if "test" not in splits:
        log("NOTE: no test split. Reported metrics come from the validation set that also selects the "
            "checkpoint, so they are optimistically biased.")

    # ---- 3. loaders ----------------------------------------------------------------------- #
    train_t, eval_t = build_transforms(cfg.img_size, cfg.channels, cfg.aug_strength)
    paths = np.array([s.path for s in samples])

    def ds(split: str, tf):
        ix = splits[split]
        return FaceDataset(paths[ix], labels[ix], tf)

    nw = min(4, os.cpu_count() or 1) if cfg.num_workers < 0 else cfg.num_workers
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    kw = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0,
              worker_init_fn=_seed_worker)
    y_tr = labels[splits["train"]]
    counts = np.bincount(y_tr, minlength=2)
    ratio = counts.min() / counts.max()
    mode = cfg.imbalance
    if mode == "auto":
        mode = "loss" if ratio < 0.8 else "none"
    log(f"Train class counts [non-ASD, ASD] = {counts.tolist()} (minority/majority {ratio:.2f}) -> imbalance handling: {mode}")
    sampler = None
    if mode == "sampler":
        w = (1.0 / counts)[y_tr]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(y_tr), replacement=True, generator=gen)
    train_loader = DataLoader(ds("train", train_t), batch_size=cfg.batch_size, shuffle=sampler is None,
                              sampler=sampler, generator=gen, drop_last=False, **kw)
    ev_bs = cfg.batch_size * 2
    val_loader = DataLoader(ds("val", eval_t), batch_size=ev_bs, shuffle=False, **kw)
    test_kw = {**kw, "persistent_workers": False}  # used once, at the end
    test_loader = DataLoader(ds("test", eval_t), batch_size=ev_bs, shuffle=False, **test_kw) if "test" in splits else None

    # ---- 4. model, optimiser ---------------------------------------------------------------- #
    model = build_model(cfg).to(device)
    gh, gw = model.token_grid(cfg.img_size, cfg.img_size)
    pc = model.count_parameters()
    log(f"\nModel: CNN {pc['cnn'] / 1e6:.2f}M + ViT {pc['vit'] / 1e6:.2f}M + head {pc['head'] / 1e3:.1f}K = "
        f"{pc['total'] / 1e6:.2f}M params | token grid {gh}x{gw} = {gh * gw} tokens | train images {len(y_tr)}")
    if gh * gw > 1500:
        log(f"WARNING: {gh * gw} tokens per image. Attention and full-resolution CNN activations get heavy: if you hit "
            f"out-of-memory, lower --batch_size, or use --downsample_factor 8 / a larger --patch_size / a smaller --img_size.")
    opt = make_optimizer(model, cfg.lr, cfg.weight_decay)

    use_amp = cfg.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    scaler = scaler_cls("cuda", enabled=use_scaler) if scaler_cls else torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if use_amp else nullcontext
    log(f"Mixed precision: {str(amp_dtype).replace('torch.', '') if use_amp else 'off'}")
    cw = None
    if mode == "loss":
        cw = torch.tensor(len(y_tr) / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps, warmup_steps = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch

    # ---- 5. training loop ---------------------------------------------------------------------- #
    monitor_key = cfg.monitor
    best_score, best_epoch, bad_epochs, step = -math.inf, 0, 0, 0
    best_thr, history = 0.5, []
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    log(f"\nTraining: up to {cfg.epochs} epochs, monitor = val {monitor_key}, patience {cfg.patience} (after warmup). "
        "Train metrics are measured on augmented batches with dropout on, so they under-read.\n")

    def save_ckpt(path: Path, epoch: int, vm: Dict, thr: float) -> None:
        torch.save({"model": model.state_dict(), "model_cfg": dataclasses.asdict(model.cfg),
                    "train_cfg": _jsonable(dataclasses.asdict(cfg)), "class_to_idx": {CLASS_NAMES[0]: 0, CLASS_NAMES[1]: 1},
                    "epoch": epoch, "val_metrics": _jsonable(vm), "val_threshold": float(thr)}, path)

    def write_log_json(extra: Optional[Dict] = None) -> None:
        payload = {"config": dataclasses.asdict(cfg), "class_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
                   "subject_scheme": scheme, "subject_level_split": subject_level, "duplicates_files": n_dup,
                   "splits": split_stats, "params": pc, "history": history,
                   "best_epoch": best_epoch, "best_score": best_score, **(extra or {})}
        with open(run_dir / "run_log.json", "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2)

    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        tm, step, lr = train_one_epoch(model, train_loader, opt, scaler, criterion, cfg, device, autocast,
                                       step, total_steps, warmup_steps, cfg.progress)
        ev = evaluate(model, val_loader, device, autocast)
        vm = compute_metrics(ev["y"], ev["p1"])
        vm["loss"] = ev["loss"]
        score = vm["auroc"] if monitor_key == "auroc" else vm["balanced_acc"]
        score = -math.inf if not math.isfinite(score) else score
        improved = score > best_score + 1e-6
        if improved:
            best_score, best_epoch, bad_epochs = score, ep, 0
            best_thr = choose_threshold(ev["y"], ev["p1"], cfg.sens_target)
            save_ckpt(best_path, ep, vm, best_thr)
        else:
            bad_epochs += 1
        save_ckpt(last_path, ep, vm, best_thr)
        history.append({"epoch": ep, "lr": lr, "train": tm, "val": vm, "seconds": time.time() - t0})
        log(f"ep {ep:03d}/{cfg.epochs} lr {lr:.2e} | train loss {tm['loss']:.4f} acc {tm['accuracy']:.3f} "
            f"bacc {tm['balanced_acc']:.3f} auroc {tm['auroc']:.3f} | val loss {vm['loss']:.4f} "
            f"{fmt_metrics(vm)} | {time.time() - t0:.0f}s{' *best*' if improved else ''}"
            + (f" | skipped {tm['skipped_batches']} non-finite batches" if tm["skipped_batches"] else ""))
        write_log_json()
        if ep > cfg.warmup_epochs and bad_epochs >= cfg.patience:
            log(f"Early stopping: no val {monitor_key} improvement for {cfg.patience} epochs.")
            break

    # ---- 6. final evaluation with the best checkpoint ---------------------------------------------- #
    if best_epoch == 0:
        raise RuntimeError("No epoch produced a finite validation score; nothing to evaluate.")
    model, ck = load_model_from_checkpoint(str(best_path), str(device))
    thr = float(ck["val_threshold"])
    ev_val = evaluate(model, val_loader, device, autocast)
    final: Dict[str, Dict] = {"val_at_0.5": compute_metrics(ev_val["y"], ev_val["p1"], 0.5),
                              "val_at_val_thr": compute_metrics(ev_val["y"], ev_val["p1"], thr)}
    write_predictions(run_dir / "val_predictions.csv", splits["val"], samples, group_names, groups, ev_val["p1"], "val", thr)
    ci = {"val": bootstrap_auroc(ev_val["y"], ev_val["p1"], groups[splits["val"]], cfg.bootstrap, cfg.seed)}
    if test_loader is not None:
        ev_te = evaluate(model, test_loader, device, autocast)
        final["test_at_0.5"] = compute_metrics(ev_te["y"], ev_te["p1"], 0.5)
        final["test_at_val_thr"] = compute_metrics(ev_te["y"], ev_te["p1"], thr)
        write_predictions(run_dir / "test_predictions.csv", splits["test"], samples, group_names, groups, ev_te["p1"], "test", thr)
        ci["test"] = bootstrap_auroc(ev_te["y"], ev_te["p1"], groups[splits["test"]], cfg.bootstrap, cfg.seed)

    rule = "highest specificity at sensitivity >= %.2f" % cfg.sens_target if cfg.sens_target else "Youden's J"
    log("\n" + "=" * 78 + "\nSUMMARY")
    log(f"Best epoch {best_epoch} (val {monitor_key} {best_score:.4f}) of {len(history)} run | "
        f"params {pc['total'] / 1e6:.2f}M | time {(time.time() - t_start) / 60:.1f} min")
    log(f"Threshold on val ({rule}): {thr:.4f}  (0.5 is NOT a calibrated clinical operating point)")
    for name in ("val_at_0.5", "val_at_val_thr", "test_at_0.5", "test_at_val_thr"):
        if name in final:
            tag = " <- threshold was tuned on this set; optimistic" if name == "val_at_val_thr" else ""
            log(f"\n{name.replace('_', ' ')}: {fmt_metrics(final[name])}{tag}\n{fmt_confusion(final[name])}")
    for split, (lo, hi) in ci.items():
        if math.isfinite(lo):
            n_g = len(set(groups[splits[split]]))
            log(f"{split} AUROC 95% bootstrap CI (resampling {n_g} {'subjects' if subject_level else 'images/groups'}): "
                f"[{lo:.3f}, {hi:.3f}]" + ("   <- too few units for this interval to mean much" if n_g < 30 else ""))
    if not subject_level:
        log("REMINDER: image-level split. Treat these numbers as an upper bound on performance for unseen children.")
    log(f"\nBest checkpoint: {best_path}\nLast checkpoint: {last_path}\nLog: {run_dir / 'run_log.json'}")
    write_log_json({"final": final, "val_threshold": thr, "threshold_rule": rule, "auroc_ci95": ci,
                    "minutes": (time.time() - t_start) / 60})
    return {"run_dir": str(run_dir), "final": final, "best_epoch": best_epoch, "val_threshold": thr}


def main(argv: Optional[List[str]] = None) -> None:
    cfg = TrainConfig(**{k: v for k, v in vars(build_parser().parse_args(argv)).items()})
    try:
        run(cfg)
    except torch.cuda.OutOfMemoryError:
        log("\nCUDA out of memory. Lower --batch_size (try 16 or 8), lower --img_size, use --downsample_factor 8, "
            "or raise --patch_size. Token count and activation size grow quickly at small strides.")
        raise


if __name__ == "__main__":
    main()
