"""
Spatially-preserving CNN feature extractor (front-end for a later ViT).

Autism vs non-autism classification pipeline: (B, C, H, W) image
-> this CNN -> (B, C_out, H', W') feature map -> [patchify + linear proj + ViT, later].

Design summary
--------------
* Stem: 3x3 conv at FULL resolution (stride 1). No aggressive 4x4/7x7-stride-2
  stem, so fine texture is seen before any resolution is thrown away.
* Stages: ConvNeXt-style residual blocks (7x7 depthwise -> LayerNorm2d ->
  1x1 expand -> GELU -> 1x1 project -> SE attention -> layer-scale -> drop-path
  -> residual add). Depthwise + pointwise keeps parameters low.
* Downsampling is explicit and controlled: exactly log2(factor) stride-2 3x3
  convs, placed at the entry of the first log2(factor) stages. Output spatial
  size is ceil(H / factor) x ceil(W / factor), so odd / non-square inputs work
  with no manual padding.
* Attention: channel SE inside every block (when enabled) + a CBAM-style
  spatial gate on the residual branch of the last block of each stage. This is
  what "selects" the features. The gate is deliberately NOT applied to the raw
  stage output: it scales all channels at a position by one scalar, and the
  next op (a channel LayerNorm) divides that scalar straight back out, leaving
  the gate with ~zero gradient (the gradient test caught exactly this).
* Normalisation: LayerNorm over channels per spatial location (LayerNorm2d).
  Independent of batch size (batch size 1 is safe) and gives the ViT tokens
  that are already channel-normalised. Computed in fp32 for fp16/bf16 safety.
* Head: 1x1 conv to `out_channels` + LayerNorm2d. Channel dim is free; project
  it to the ViT embedding dim later with a Linear/1x1 conv.
* Never flattens. Fully convolutional, so any H, W >= 1 is accepted. The ViT
  positional embedding must be interpolated if train/test grids differ.
* Determinism: global pooling is done with .mean() rather than
  adaptive_avg_pool2d (whose CUDA backward is nondeterministic). Everything
  else is conv / elementwise / amax, which are deterministic under
  torch.backends.cudnn.deterministic = True.

Why a CNN before the ViT (and the honest caveat)
------------------------------------------------
Raw ViT patching (e.g. 16x16 linear patch embed) gives every token a single
linear view of 768 raw pixels: no locality prior, no translation equivariance,
and edges / texture are only recoverable if the transformer learns them from
data. Autism-screening face datasets are small (low thousands of images), the
regime where ViTs are data-starved and CNN inductive bias helps. The CNN learns
local filters first and hands the ViT tokens that already encode them.
Caveat: that is the standard argument, not something this file proves. A CNN
trained from scratch on a few thousand images can still overfit badly; you
need to compare against a pretrained backbone and report subject-level splits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class CNNExtractorConfig:
    in_channels: int = 3                 # 1 (grayscale) or 3 (RGB)
    out_channels: int = 256              # channels of the output feature map
    num_stages: int = 3                  # number of stages (>= log2(downsample_factor))
    depth: Union[int, Sequence[int]] = 2  # blocks per stage (int or one per stage)
    base_width: int = 48                 # channels of stem / stage 0
    width_mult: float = 2.0              # width multiplier per stage
    downsample_factor: int = 8           # total spatial reduction: 2, 4 or 8
    use_attention: bool = True           # SE in blocks + spatial gate in last block per stage
    expansion: int = 4                   # inverted-bottleneck expansion ratio
    kernel_size: int = 7                 # depthwise kernel size (odd)
    se_ratio: int = 4                    # SE channel reduction ratio
    layer_scale_init: float = 1e-2       # residual branch scale init
    drop_path_rate: float = 0.1          # max stochastic-depth rate
    auto_adapt_channels: bool = True     # 1<->3 channel input adaptation in forward

    def __post_init__(self) -> None:
        if self.in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        if self.downsample_factor not in (2, 4, 8):
            raise ValueError("downsample_factor must be 2, 4 or 8")
        n_down = int(math.log2(self.downsample_factor))
        if self.num_stages < n_down:
            raise ValueError(
                f"num_stages={self.num_stages} < log2(downsample_factor)={n_down}; "
                "need at least one stage per stride-2 step"
            )
        if isinstance(self.depth, int):
            self.depth = tuple([self.depth] * self.num_stages)
        else:
            self.depth = tuple(self.depth)
        if len(self.depth) != self.num_stages or min(self.depth) < 1:
            raise ValueError("depth must be an int >= 1 or one int >= 1 per stage")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

    @property
    def n_downsamples(self) -> int:
        return int(math.log2(self.downsample_factor))

    @property
    def dims(self) -> Tuple[int, ...]:
        return tuple(int(round(self.base_width * self.width_mult ** i)) for i in range(self.num_stages))


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim at each spatial location, for NCHW tensors.

    Statistics are computed in fp32 so fp16/bf16 autocast is safe. Output is
    cast back to the input dtype.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.float()
        mean = u.mean(dim=1, keepdim=True)
        var = (u - mean).pow(2).mean(dim=1, keepdim=True)
        y = (u - mean) * torch.rsqrt(var + self.eps)
        y = y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y.to(x.dtype)


class SqueezeExcite(nn.Module):
    """Channel attention. Global pooling via .mean() (deterministic backward)."""

    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        mid = max(4, channels // ratio)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = self.fc2(F.silu(self.fc1(s)))
        return x * torch.sigmoid(s)


class SpatialAttention(nn.Module):
    """CBAM-style spatial gate: where in the map to look. Keeps shape (B,C,H,W).
    Use on a residual branch, not before a channel LayerNorm (it would cancel)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth per sample. Identity in eval mode."""
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class Block(nn.Module):
    """ConvNeXt-style residual block with optional SE attention."""

    def __init__(self, dim: int, expansion: int, kernel_size: int, use_attention: bool,
                 se_ratio: int, layer_scale_init: float, dp: float,
                 spatial_gate: bool = False) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, 1)
        self.pw2 = nn.Conv2d(dim * expansion, dim, 1)
        self.se = SqueezeExcite(dim, se_ratio) if use_attention else nn.Identity()
        # Spatial gate lives on the residual branch (see module docstring: a gate
        # on the stage output would be cancelled by the next channel LayerNorm).
        self.sa = SpatialAttention() if (use_attention and spatial_gate) else nn.Identity()
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.dp = dp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(self.dw(x))
        x = self.pw2(F.gelu(self.pw1(x)))
        x = self.sa(self.se(x))
        x = x * self.gamma.view(1, -1, 1, 1).to(x.dtype)
        return r + drop_path(x, self.dp, self.training)


class Stage(nn.Module):
    def __init__(self, c_in: int, c_out: int, depth: int, downsample: bool,
                 cfg: CNNExtractorConfig, dps: Sequence[float]) -> None:
        super().__init__()
        if downsample:
            # 3x3 stride-2, pad 1 -> output = ceil(H/2): odd sizes need no padding.
            self.transition: nn.Module = nn.Sequential(
                LayerNorm2d(c_in), nn.Conv2d(c_in, c_out, 3, stride=2, padding=1))
        elif c_in != c_out:
            self.transition = nn.Sequential(LayerNorm2d(c_in), nn.Conv2d(c_in, c_out, 1))
        else:
            self.transition = nn.Identity()
        self.blocks = nn.Sequential(*[
            Block(c_out, cfg.expansion, cfg.kernel_size, cfg.use_attention,
                  cfg.se_ratio, cfg.layer_scale_init, dps[i],
                  spatial_gate=(i == depth - 1))
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.transition(x))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SpatialCNNExtractor(nn.Module):
    """(B, C, H, W) -> (B, out_channels, ceil(H/f), ceil(W/f)). Never flattens."""

    def __init__(self, cfg: Optional[CNNExtractorConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CNNExtractorConfig()
        cfg = self.cfg
        dims = cfg.dims

        # Full-resolution stem: no information discarded before the first stage.
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, dims[0], 3, padding=1),
            LayerNorm2d(dims[0]),
            nn.GELU(),
        )

        total = sum(cfg.depth)
        dpr = [cfg.drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, k, c_prev = [], 0, dims[0]
        for i in range(cfg.num_stages):
            stages.append(Stage(c_prev, dims[i], cfg.depth[i], i < cfg.n_downsamples,
                                cfg, dpr[k:k + cfg.depth[i]]))
            k += cfg.depth[i]
            c_prev = dims[i]
        self.stages = nn.Sequential(*stages)

        # Clean channel dim + per-token channel normalisation for the ViT.
        self.head = nn.Sequential(nn.Conv2d(dims[-1], cfg.out_channels, 1),
                                  LayerNorm2d(cfg.out_channels))
        self.apply(self._init_weights)
        self._init_dense_convs()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_dense_convs(self) -> None:
        # Dense (non-depthwise) convs that sit on the main path get He init so
        # activations don't start vanishing; block convs keep small init and
        # rely on layer-scale + residual identity for stable early training.
        convs = [self.stem[0]]
        for s in self.stages:
            if isinstance(s.transition, nn.Sequential):
                convs.append(s.transition[1])
        for c in convs:
            nn.init.kaiming_normal_(c.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(c.bias)
        # Block MLP and SE convs: trunc_normal(0.02) made the SE weights receive
        # ~1e-10 gradients at init (tiny pooled input x layer-scale 0.01), i.e.
        # they would not learn under plain SGD. He init on the 1x1 convs fixes
        # that; the residual branch stays small because of layer-scale (gamma).
        for m in self.modules():
            if isinstance(m, Block):
                mods = [m.pw1, m.pw2] + ([m.se.fc1, m.se.fc2]
                                         if isinstance(m.se, SqueezeExcite) else [])
                for c in mods:
                    nn.init.kaiming_normal_(c.weight, mode="fan_in", nonlinearity="relu")
                    nn.init.zeros_(c.bias)

    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        c, want = x.shape[1], self.cfg.in_channels
        if c == want:
            return x
        if not self.cfg.auto_adapt_channels:
            raise ValueError(f"expected {want} channels, got {c}")
        if c == 1 and want == 3:
            return x.expand(-1, 3, -1, -1)
        if c == 3 and want == 1:
            w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            return (x * w).sum(dim=1, keepdim=True)
        raise ValueError(f"cannot adapt {c} channels to {want}")

    def output_size(self, h: int, w: int) -> Tuple[int, int]:
        f = self.cfg.downsample_factor
        return math.ceil(h / f), math.ceil(w / f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or min(x.shape[2:]) < 1:
            raise ValueError(f"expected (B, C, H, W) with H, W >= 1, got {tuple(x.shape)}")
        x = self._adapt_channels(x)
        return self.head(self.stages(self.stem(x)))


# --------------------------------------------------------------------------- #
# ViT hand-off helper
# --------------------------------------------------------------------------- #
def feature_map_to_tokens(fmap: torch.Tensor, patch_size: int = 1
                          ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """(B, C, H', W') -> (B, N, C * p * p) tokens plus the (gh, gw) token grid.

    patch_size=1: every CNN cell is one token (most common for CNN-ViT hybrids).
    patch_size>1: p x p cells are merged into one token (fewer tokens).
    Bottom/right edge is replicate-padded if H' or W' is not divisible by p.
    Keep (gh, gw) to interpolate positional embeddings at other resolutions.
    """
    b, c, h, w = fmap.shape
    p = patch_size
    if p < 1:
        raise ValueError("patch_size must be >= 1")
    pad_h, pad_w = (-h) % p, (-w) % p
    if pad_h or pad_w:
        fmap = F.pad(fmap, (0, pad_w, 0, pad_h), mode="replicate")
        h, w = h + pad_h, w + pad_w
    if p == 1:
        tokens = fmap.flatten(2).transpose(1, 2)
    else:
        tokens = F.unfold(fmap, kernel_size=p, stride=p).transpose(1, 2)
    return tokens, (h // p, w // p)


# --------------------------------------------------------------------------- #
# Tests + usage example
# --------------------------------------------------------------------------- #
def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def run_tests() -> None:
    torch.manual_seed(0)

    # 1. Shape consistency: factors x channels x awkward sizes x batch size 1
    cases = [(1, 64, 64), (1, 97, 131), (2, 200, 320), (1, 384, 384), (1, 512, 512)]
    for factor in (2, 4, 8):
        for in_ch in (1, 3):
            model = SpatialCNNExtractor(CNNExtractorConfig(
                in_channels=in_ch, downsample_factor=factor)).eval()
            for b, h, w in cases:
                if factor == 2 and max(h, w) > 200:
                    continue  # full-res-heavy on CPU; covered at factor 4/8
                with torch.no_grad():
                    y = model(torch.randn(b, in_ch, h, w))
                exp = (b, model.cfg.out_channels, *model.output_size(h, w))
                assert tuple(y.shape) == exp, (factor, in_ch, (b, h, w), tuple(y.shape), exp)
                assert torch.isfinite(y).all()
    print("[ok] shapes: factors 2/4/8, 1 & 3 channels, non-square, 64..512, B=1")

    # 1b. Channel adaptation: 3-ch model fed grayscale, 1-ch model fed RGB
    m3 = SpatialCNNExtractor(CNNExtractorConfig(in_channels=3, downsample_factor=4)).eval()
    m1 = SpatialCNNExtractor(CNNExtractorConfig(in_channels=1, downsample_factor=4)).eval()
    with torch.no_grad():
        assert m3(torch.randn(1, 1, 64, 64)).shape[1] == 256
        assert m1(torch.randn(1, 3, 64, 64)).shape[1] == 256
    print("[ok] channel adaptation 1<->3")

    # 2. Gradient flow (train mode, batch size 1, odd size)
    model = SpatialCNNExtractor(CNNExtractorConfig(downsample_factor=8)).train()
    y = model(torch.randn(1, 3, 97, 131))
    # Not y.pow(2).mean(): the final LayerNorm makes that nearly constant.
    (y * torch.randn_like(y)).mean().backward()
    norms = {n: p.grad.norm().item() for n, p in model.named_parameters()
             if p.grad is not None}
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no grad: {missing}"
    assert all(math.isfinite(v) for v in norms.values())
    nz = [v for v in norms.values() if v > 0]
    gate = [v for n, v in norms.items() if ".sa." in n]
    assert gate and min(gate) > 1e-6, f"spatial gate gradients dead: {gate}"
    assert min(nz) > 1e-8, f"vanishing gradient: min nonzero norm {min(nz):.2e}"
    print(f"[ok] gradients: {len(norms)} params, all finite, "
          f"min nonzero norm {min(nz):.2e}, max {max(norms.values()):.2e}, "
          f"zero-grad params: {len(norms) - len(nz)}")

    # 3. Determinism: same seed -> identical weights -> identical eval output
    x = torch.randn(2, 3, 128, 160)
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        m = SpatialCNNExtractor().eval()
        with torch.no_grad():
            outs.append(m(x))
    assert torch.equal(outs[0], outs[1])
    print("[ok] deterministic init + eval forward")

    # 4. Mixed precision
    m = SpatialCNNExtractor().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        yb = m(torch.randn(2, 3, 128, 128))
    assert torch.isfinite(yb.float()).all()
    print(f"[ok] bf16 autocast (CPU), output dtype {yb.dtype}")
    if torch.cuda.is_available():
        m = m.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            yh = m(torch.randn(2, 3, 128, 128, device="cuda"))
        assert torch.isfinite(yh.float()).all()
        print("[ok] fp16 autocast (CUDA)")
    else:
        print("[skip] fp16 CUDA autocast: no GPU in this environment")

    # 5. Patch helper
    fmap = torch.randn(2, 256, 13, 17)
    t1, g1 = feature_map_to_tokens(fmap, 1)
    t2, g2 = feature_map_to_tokens(fmap, 2)
    assert t1.shape == (2, 13 * 17, 256) and g1 == (13, 17)
    assert t2.shape == (2, 7 * 9, 256 * 4) and g2 == (7, 9)
    print("[ok] feature_map_to_tokens p=1 and p=2 (with padding)")

    # 6. Attention off still works
    m = SpatialCNNExtractor(CNNExtractorConfig(use_attention=False)).eval()
    with torch.no_grad():
        m(torch.randn(1, 3, 64, 64))
    print("[ok] use_attention=False")


def main() -> None:
    # ---- usage example ----
    cfg = CNNExtractorConfig(in_channels=3, out_channels=256, num_stages=3,
                             depth=2, downsample_factor=8, use_attention=True)
    cnn = SpatialCNNExtractor(cfg)
    x = torch.randn(4, 3, 224, 224)
    fmap = cnn(x)
    tokens, grid = feature_map_to_tokens(fmap, patch_size=1)
    print(f"params: {_count_params(cnn) / 1e6:.2f}M")
    print(f"input {tuple(x.shape)} -> feature map {tuple(fmap.shape)} "
          f"-> tokens {tuple(tokens.shape)} (grid {grid})")
    print("--- tests ---")
    run_tests()


if __name__ == "__main__":
    main()
