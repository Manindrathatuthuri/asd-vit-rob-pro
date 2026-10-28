"""
CNN -> ViT classifier for ASD facial-image screening (binary: Autistic vs Non-Autistic).

Pipeline
--------
image (B,C,H,W) -> SpatialCNNExtractor -> fmap (B,C_out,H',W')
  -> feature_map_to_tokens (imported, not re-implemented) -> tokens (B,N,C_out*p^2)
  -> Linear projection to embed_dim -> + 2-D-interpolable positional embedding
  -> [CLS] + pre-norm Transformer blocks -> CLS -> (LayerNorm) -> Linear -> 2 logits

`cnn_feature_extractor.py` is imported and used as is. Its weights keep the
CNN's own initialisation; only the ViT and head are initialised here.

Class convention (matters for every metric you compute later)
-------------------------------------------------------------
index 0 = Non-Autistic, index 1 = Autistic. `probs[:, 1]` is P(ASD), the positive
class for sensitivity / AUROC. Train with nn.CrossEntropyLoss on the 2 logits.

Hyper-parameters and recommended starting defaults
--------------------------------------------------
These are conventional small-data ViT settings, NOT tuned on any ASD dataset.
Treat them as a starting point for a sweep, not as validated.

ViTConfig
  embed_dim=256        Token width. Equal to the CNN's default out_channels (256).
  depth=6              Transformer blocks. ~0.79M params per block at 256 wide.
                       For a few thousand images try 4 before going deeper.
  num_heads=4          Must divide embed_dim (head dim 64).
  mlp_ratio=4.0        MLP hidden = mlp_ratio * embed_dim.
  drop_rate=0.1        Dropout on embeddings, attention output and MLP.
  attn_drop_rate=0.1   Dropout on attention weights.
  drop_path_rate=0.1   Max stochastic depth, linearly ramped 0 -> rate over blocks.
  patch_size=1         CNN cells merged per token side. 1 = one token per cell.
                       Use >= 2 whenever the CNN downsample factor is 2 or 4.
  head_norm=True       LayerNorm on the CLS vector before the classifier.
  qkv_bias=True        Bias in the qkv projection.
  max_tokens=4096      Hard cap on token count (attention is O(N^2) in memory).
                       Raises a clear error instead of an OOM. None disables.
  deterministic_attention=False
                       True forces the plain "math" SDPA kernel. Use it if you need
                       bit-reproducible training: the fused flash / mem-efficient
                       attention backward can be nondeterministic on CUDA, and
                       torch.backends.cudnn.deterministic does not cover it.
ASDModelConfig
  cnn                  CNNExtractorConfig (ignored if you pass a ready-made CNN).
  vit                  ViTConfig.
  img_size=224         Training resolution. Fixes the *reference* token grid the
                       positional embedding is stored at; other resolutions are
                       handled by interpolation.

Positional embeddings
---------------------
Stored as a learnable 1-D sequence (1, gh0*gw0, D) for the reference grid. At a
different grid size it is reshaped to 2-D and resized (bicubic, fp32) to the new
grid. The CLS token has no positional embedding (it is not a place in the image).
Caveat: if the grid differs at *training* time, the bicubic backward on CUDA is
nondeterministic. Train at one grid size, interpolate only at inference.

Subgroup evaluation
-------------------
Nothing in the model mixes samples across the batch (LayerNorm only, no
BatchNorm), so a sample's output does not depend on who else is in the batch;
this is tested. `forward_features` returns the CLS embedding and `predict` the
probabilities, so you can slice any batch by an external subgroup label.
The model does nothing about bias itself.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cnn_feature_extractor import (  # CNN is used unchanged
    CNNExtractorConfig,
    SpatialCNNExtractor,
    drop_path,
    feature_map_to_tokens,
)

NUM_CLASSES = 2
NON_AUTISTIC, AUTISTIC = 0, 1
CLASS_NAMES = ("non_autistic", "autistic")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ViTConfig:
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1
    patch_size: int = 1
    head_norm: bool = True
    qkv_bias: bool = True
    max_tokens: Optional[int] = 4096
    deterministic_attention: bool = False

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim={self.embed_dim} not divisible by num_heads={self.num_heads}")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        for name in ("drop_rate", "attn_drop_rate", "drop_path_rate"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 or None")


@dataclass
class ASDModelConfig:
    cnn: CNNExtractorConfig = field(default_factory=CNNExtractorConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    img_size: Union[int, Tuple[int, int]] = 224  # training resolution (H, W)

    def __post_init__(self) -> None:
        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size)
        self.img_size = (int(self.img_size[0]), int(self.img_size[1]))
        if min(self.img_size) < 1:
            raise ValueError("img_size must be positive")


# --------------------------------------------------------------------------- #
# ViT components
# --------------------------------------------------------------------------- #
class PosEmbed2D(nn.Module):
    """Learnable positional embedding stored 1-D, resizable as a 2-D grid."""

    def __init__(self, ref_grid: Tuple[int, int], dim: int) -> None:
        super().__init__()
        self.ref_grid = ref_grid
        self.pos = nn.Parameter(torch.zeros(1, ref_grid[0] * ref_grid[1], dim))

    def forward(self, grid: Tuple[int, int]) -> torch.Tensor:
        """Return (1, gh*gw, D) for the requested grid (fp32)."""
        if tuple(grid) == tuple(self.ref_grid):
            return self.pos
        gh0, gw0 = self.ref_grid
        p = self.pos.float().reshape(1, gh0, gw0, -1).permute(0, 3, 1, 2)
        # antialias matters when shrinking the grid (e.g. 28x28 -> 20x20).
        p = F.interpolate(p, size=tuple(grid), mode="bicubic", align_corners=False, antialias=True)
        return p.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1)


class Attention(nn.Module):
    """Multi-head self-attention on top of fused scaled_dot_product_attention."""

    def __init__(self, dim: int, heads: int, qkv_bias: bool, attn_drop: float,
                 proj_drop: float, deterministic: bool) -> None:
        super().__init__()
        self.heads = heads
        self.attn_drop = attn_drop
        self.deterministic = deterministic
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # Attention dropout must be off in eval: gate on self.training explicitly.
        p = self.attn_drop if self.training else 0.0
        ctx = sdpa_kernel(SDPBackend.MATH) if self.deterministic else nullcontext()
        with ctx:
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        return self.proj_drop(self.proj(o.transpose(1, 2).reshape(b, n, c)))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-norm: x + DP(Attn(LN(x))); x + DP(MLP(LN(x)))."""

    def __init__(self, cfg: ViTConfig, dp: float) -> None:
        super().__init__()
        d = cfg.embed_dim
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.attn = Attention(d, cfg.num_heads, cfg.qkv_bias, cfg.attn_drop_rate,
                              cfg.drop_rate, cfg.deterministic_attention)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.mlp = MLP(d, int(d * cfg.mlp_ratio), cfg.drop_rate)
        self.dp = dp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + drop_path(self.attn(self.norm1(x)), self.dp, self.training)
        return x + drop_path(self.mlp(self.norm2(x)), self.dp, self.training)


class ViTEncoder(nn.Module):
    """tokens (B,N,token_dim) + their (gh,gw) grid -> CLS embedding (B,D)."""

    def __init__(self, cfg: ViTConfig, token_dim: int, ref_grid: Tuple[int, int]) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.token_proj = nn.Linear(token_dim, d)       # CNN channels -> ViT width
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = PosEmbed2D(ref_grid, d)
        self.pos_drop = nn.Dropout(cfg.drop_rate)
        dpr = [cfg.drop_path_rate * i / max(cfg.depth - 1, 1) for i in range(cfg.depth)]
        self.blocks = nn.ModuleList([TransformerBlock(cfg, dpr[i]) for i in range(cfg.depth)])
        self.apply(_init_vit_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.pos, std=0.02)

    def check_num_tokens(self, n: int) -> None:
        if self.cfg.max_tokens is not None and n > self.cfg.max_tokens:
            raise ValueError(
                f"{n} tokens exceeds max_tokens={self.cfg.max_tokens} (attention memory is "
                f"O(N^2)). Increase the CNN downsample_factor, increase ViTConfig.patch_size, "
                f"or set max_tokens=None."
            )

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        b, n, d_in = tokens.shape
        if n != grid[0] * grid[1]:
            raise ValueError(f"{n} tokens do not match grid {tuple(grid)}")
        if d_in != self.token_proj.in_features:
            raise ValueError(f"token dim {d_in} != expected {self.token_proj.in_features}")
        self.check_num_tokens(n)
        x = self.token_proj(tokens)
        x = x + self.pos_embed(grid).to(x.dtype)        # CLS gets no positional embedding
        x = torch.cat([self.cls_token.to(x.dtype).expand(b, -1, -1), x], dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x[:, 0]


class ClassificationHead(nn.Module):
    """CLS vector -> (LayerNorm) -> Linear -> 2 logits.

    The LayerNorm is applied to the CLS vector only: LayerNorm is per-token, so
    this equals the usual 'final norm over all tokens, then take CLS' at lower cost.
    """

    def __init__(self, dim: int, head_norm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6) if head_norm else nn.Identity()
        self.fc = nn.Linear(dim, NUM_CLASSES)
        self.apply(_init_vit_weights)

    def forward(self, cls_vec: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls_vec))


def _init_vit_weights(m: nn.Module) -> None:
    """Truncated-normal Linear, identity-initialised LayerNorm."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class CNNVITASDClassifier(nn.Module):
    """SpatialCNNExtractor -> tokens -> ViT -> 2 logits (Non-Autistic, Autistic)."""

    NON_AUTISTIC = NON_AUTISTIC
    AUTISTIC = AUTISTIC

    def __init__(self, cfg: Optional[ASDModelConfig] = None,
                 cnn: Optional[SpatialCNNExtractor] = None) -> None:
        super().__init__()
        cfg = cfg or ASDModelConfig()
        # A supplied CNN wins; keep the stored config consistent with it.
        self.cnn = cnn if cnn is not None else SpatialCNNExtractor(cfg.cnn)
        self.cfg = replace(cfg, cnn=self.cnn.cfg)

        p = cfg.vit.patch_size
        h0, w0 = self.cnn.output_size(*cfg.img_size)
        ref_grid = (math.ceil(h0 / p), math.ceil(w0 / p))
        self.vit = ViTEncoder(cfg.vit, self.cnn.cfg.out_channels * p * p, ref_grid)
        self.head = ClassificationHead(cfg.vit.embed_dim, cfg.vit.head_norm)

    # -- helpers ---------------------------------------------------------- #
    def token_grid(self, h: int, w: int) -> Tuple[int, int]:
        """Token grid (gh, gw) produced for an (h, w) input, without running the CNN."""
        p = self.cfg.vit.patch_size
        hp, wp = self.cnn.output_size(h, w)
        return math.ceil(hp / p), math.ceil(wp / p)

    def count_parameters(self) -> dict:
        n = lambda m: sum(q.numel() for q in m.parameters())
        return {"cnn": n(self.cnn), "vit": n(self.vit), "head": n(self.head), "total": n(self)}

    # -- forward ---------------------------------------------------------- #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> CLS embedding (B, embed_dim). Use for subgroup / embedding analysis."""
        if x.ndim == 4:  # fail on token count before spending CNN compute
            gh, gw = self.token_grid(x.shape[2], x.shape[3])
            self.vit.check_num_tokens(gh * gw)
        fmap = self.cnn(x)
        tokens, grid = feature_map_to_tokens(fmap, self.cfg.vit.patch_size)
        return self.vit(tokens, grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns logits (B, 2)
        return self.head(self.forward_features(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(probabilities (B,2), predicted class (B,)). Always runs in eval mode and
        restores each submodule's previous train/eval flag afterwards."""
        flags = [(m, m.training) for m in self.modules()]
        self.eval()
        try:
            probs = F.softmax(self.forward(x).float(), dim=-1)
        finally:
            for m, t in flags:
                m.training = t
        return probs, probs.argmax(dim=-1)


# --------------------------------------------------------------------------- #
# Tests + usage example
# --------------------------------------------------------------------------- #
def _build(factor: int = 8, patch: int = 1, **vit_kw) -> CNNVITASDClassifier:
    return CNNVITASDClassifier(ASDModelConfig(
        cnn=CNNExtractorConfig(downsample_factor=factor),
        vit=ViTConfig(patch_size=patch, **vit_kw), img_size=224))


def run_tests() -> None:
    torch.manual_seed(0)

    # 1. Shapes: resolutions x downsample factors x patch sizes, B=1 and B>1, non-square, odd.
    plan = {  # (factor, patch) -> sizes (CPU-feasible; factor 2 is full-res heavy)
        (8, 1): [(160, 160), (224, 224), (384, 384), (200, 312), (97, 131)],
        (8, 2): [(224, 224), (100, 100)],
        (4, 2): [(160, 160), (224, 224), (100, 100)],
        (2, 4): [(96, 96), (128, 160)],
    }
    for (f, p), sizes in plan.items():
        m = _build(f, p).eval()
        for h, w in sizes:
            for b in (1, 2):
                with torch.no_grad():
                    y = m(torch.randn(b, 3, h, w))
                assert tuple(y.shape) == (b, 2) and torch.isfinite(y).all(), (f, p, h, w, b)
        print(f"[ok] shapes factor={f} patch={p}: {sizes} (B=1,2) -> grid ref {m.vit.pos_embed.ref_grid}")

    # 2. Positional-embedding interpolation.
    pe = PosEmbed2D((28, 28), 8)
    nn.init.trunc_normal_(pe.pos, std=1.0)
    assert torch.equal(pe((28, 28)), pe.pos)
    with torch.no_grad():
        pe.pos.fill_(0.7)
        for g in [(20, 20), (48, 48), (13, 39), (7, 7)]:
            out = pe(g)
            assert out.shape == (1, g[0] * g[1], 8)
            assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-5), g
        ramp = torch.linspace(0, 1, 28).view(1, 1, 28).expand(1, 28, 28)
        pe.pos.copy_(ramp.reshape(1, 784, 1).expand(1, 784, 8))
        row = pe((28, 56)).reshape(28, 56, 8)[14, :, 0]
        corr = torch.corrcoef(torch.stack([row, torch.linspace(0, 1, 56)]))[0, 1].item()
        assert corr > 0.999, corr
    print("[ok] pos-embed: identity at ref grid, constants preserved, ramp preserved (corr>0.999), "
          "non-square / up / down")

    # 3. Train vs eval: no leakage.
    m = _build(drop_rate=0.3, attn_drop_rate=0.3, drop_path_rate=0.5)
    x = torch.randn(2, 3, 224, 224)
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b), "eval must be deterministic"
    m.train()
    with torch.no_grad():
        c, d = m(x), m(x)
    assert not torch.equal(c, d), "train-mode regularisers appear inactive"
    print("[ok] eval deterministic; train-mode dropout / drop-path active "
          f"(train-vs-train max diff {(c - d).abs().max().item():.2e})")

    # 4. Batch independence (what makes per-subgroup evaluation trustworthy).
    m.eval()
    xb = torch.randn(3, 3, 160, 160)
    with torch.no_grad():
        yb = m(xb)
        ys = torch.cat([m(xb[i:i + 1]) for i in range(3)])
    diff = (yb - ys).abs().max().item()
    assert diff < 1e-4, diff
    print(f"[ok] batch independence: batch-of-3 vs singles max diff {diff:.2e}")

    # 5. Gradient flow, end to end through the CNN.
    m = _build().train()
    y = m(torch.randn(2, 3, 224, 224))
    (y * torch.randn_like(y)).sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    norms = {n: p.grad.norm().item() for n, p in m.named_parameters()}
    assert all(math.isfinite(v) for v in norms.values())
    lo = min(norms, key=norms.get)
    print(f"[ok] gradients reach all {len(norms)} params (CNN included); "
          f"min norm {norms[lo]:.2e} ({lo}), max {max(norms.values()):.2e}; "
          f"zero-grad: {sum(v == 0 for v in norms.values())}")

    # 6. Mixed precision.
    for f, p, s in [(8, 1, 224), (4, 2, 100)]:
        mm = _build(f, p).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            yb16 = mm(torch.randn(2, 3, s, s))
        assert torch.isfinite(yb16.float()).all()
    print(f"[ok] bf16 autocast (CPU) incl. odd grid with padding; logits dtype {yb16.dtype}")
    if torch.cuda.is_available():
        mm = _build().eval().cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            assert torch.isfinite(mm(torch.randn(2, 3, 224, 224, device="cuda")).float()).all()
        print("[ok] fp16 autocast (CUDA)")
    else:
        print("[skip] fp16 CUDA autocast and CUDA determinism: no GPU in this environment")

    # 7. predict(): probabilities, classes, mode restoration.
    m = _build().train()
    m.cnn.eval()  # a deliberately mixed state must be restored exactly
    probs, pred = m.predict(torch.randn(4, 3, 224, 224))
    assert probs.shape == (4, 2) and pred.shape == (4,)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.equal(pred, probs.argmax(-1))
    assert m.training and m.vit.training and not m.cnn.training
    print("[ok] predict(): probs sum to 1, classes = argmax, train/eval flags restored")

    # 8. Pre-instantiated CNN: used as is; grayscale CNN fed RGB; weights untouched.
    cnn = SpatialCNNExtractor(CNNExtractorConfig(in_channels=1, downsample_factor=4, out_channels=128))
    before = sum(p.double().sum().item() for p in cnn.parameters())
    m = CNNVITASDClassifier(ASDModelConfig(vit=ViTConfig(patch_size=2)), cnn=cnn).eval()
    assert m.cnn is cnn and m.cfg.cnn is cnn.cfg
    assert abs(before - sum(p.double().sum().item() for p in cnn.parameters())) < 1e-9
    with torch.no_grad():
        assert m(torch.randn(1, 3, 128, 128)).shape == (1, 2)
    print("[ok] pre-built CNN accepted unchanged (1-ch CNN fed RGB, 128 channels, patch 2)")

    # 9. Token-count guard: fails early and clearly instead of OOM.
    m = _build(4, 1).eval()
    try:
        m(torch.randn(1, 3, 384, 384))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "max_tokens" in str(e)
    print("[ok] token guard: factor 4 / patch 1 at 384px (9216 tokens) raises ValueError before the CNN runs")

    # 10. Deterministic-attention path matches the default kernel.
    m1, m2 = _build().eval(), _build(deterministic_attention=True).eval()
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        d = (m1(x) - m2(x)).abs().max().item()
    assert d < 1e-5, d
    print(f"[ok] deterministic_attention=True matches default kernel (max diff {d:.1e})")

    # 11. Config validation.
    for bad in (dict(embed_dim=250, num_heads=4), dict(depth=0), dict(drop_path_rate=1.0)):
        try:
            ViTConfig(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    print("[ok] config validation")


def main() -> None:
    # ---- usage example ----
    model = CNNVITASDClassifier(ASDModelConfig(
        cnn=CNNExtractorConfig(in_channels=3, out_channels=256, downsample_factor=8),
        vit=ViTConfig(embed_dim=256, depth=6, num_heads=4, patch_size=1),
        img_size=224,
    ))
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)                       # training-style forward: (B, 2)
    probs, pred = model.predict(x)          # eval-mode probabilities and classes
    cnt = model.count_parameters()
    print(f"params: cnn {cnt['cnn'] / 1e6:.2f}M | vit {cnt['vit'] / 1e6:.2f}M | "
          f"head {cnt['head'] / 1e3:.1f}K | total {cnt['total'] / 1e6:.2f}M")
    print(f"logits {tuple(logits.shape)} | probs {tuple(probs.shape)} | pred {pred.tolist()} "
          f"({', '.join(CLASS_NAMES[i] for i in pred.tolist())})")
    print(f"P(autistic) = {[round(v, 3) for v in probs[:, AUTISTIC].tolist()]}")
    print("--- tests ---")
    run_tests()


if __name__ == "__main__":
    main()
