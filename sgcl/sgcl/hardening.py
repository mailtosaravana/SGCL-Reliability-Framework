"""Architecture-level reliability enhancement (Section III-G):
   (1) selective reinforcement: duplicate-and-compare execution of the critical layers, recomputation on disagreement;
   (2) activation-range restriction: per-layer clip at clip_factor x the 99.99th percentile of clean activations
       (calib_images training images), logits clipped to +/- logit_clip;
   (3) residual-path regularisation beta on layer3/layer4 is a training-time term and is implemented in common.train().
Layer-wise sensitivity estimation (Section III-F) is also here."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np, torch, torch.nn as nn
from .common import DEVICE, injectable_layers, block_of, monte_carlo, accuracy

# ----------------------------------------------------------------------------- sensitivity (III-F)
def layer_sensitivity(model, dl, lam: float, n_trials: int, clean_acc: float, blocks: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """S_k = E[CER_k] with single-bit activation corruption restricted to block k (Eq. 7). Returned in percentage points."""
    layers = injectable_layers(model)
    groups: Dict[str, List[str]] = {}
    for n in layers: groups.setdefault(block_of(n), []).append(n)
    if blocks: groups = {k: v for k, v in groups.items() if k in blocks}
    S = {}
    for k, names in groups.items():
        r = monte_carlo(model, dl, lam, n_trials, clean_acc, layers=names, weights=False, activations=True, log_every=0, desc=k)
        S[k] = 100.0 * r["cer_mean"]; print(f"   sensitivity {k:12s} S_k = {S[k]:.3f} pp", flush=True)
    return S

def critical_set(S: Dict[str, float], T_pp: float, rule: str = "threshold") -> Tuple[List[str], float]:
    """rule = "threshold": C = {k : S_k >= T_pp} (fixed threshold).
       rule = "median":    T is set to the median of the S_k so that the upper half of the sensitivity ranking is critical
                           (used when the fixed threshold is not resolvable at the operating point). Returns (C, T used)."""
    if rule == "median":
        T_pp = float(np.median(list(S.values())))
    return [k for k, v in S.items() if v >= T_pp], T_pp

# ----------------------------------------------------------------------------- activation-range restriction
@torch.no_grad()
def calibrate_bounds(model, calib_dl, n_images: int, percentile: float, factor: float) -> Dict[str, float]:
    """Per-layer bound = factor x percentile of |clean output| over n_images training images (Table 2)."""
    model.eval(); layers = injectable_layers(model); samples = {n: [] for n in layers}; hs = []
    for n, m in layers.items():
        def hook(mod, inp, out, name=n):
            a = out.detach().abs().flatten()
            if a.numel() > 100_000: a = a[torch.randint(0, a.numel(), (100_000,), device=a.device)]
            samples[name].append(a.float().cpu())
        hs.append(m.register_forward_hook(hook))
    seen = 0
    for x, _ in calib_dl:
        model(x.to(DEVICE)); seen += x.shape[0]
        if seen >= n_images: break
    for h in hs: h.remove()
    return {n: float(factor * torch.quantile(torch.cat(v), percentile / 100.0)) for n, v in samples.items()}

class RangeRestriction:
    """Context manager: clip every conv/linear output to [-b_n, b_n] and the logits to [-c, c]."""
    def __init__(self, model, bounds: Dict[str, float], logit_clip: float):
        self.model, self.bounds, self.logit_clip, self.handles = model, bounds, logit_clip, []
    def __enter__(self):
        for n, m in injectable_layers(self.model).items():
            b = self.bounds.get(n, self.bounds.get(n[:-6] if n.endswith(".inner") else n))   # DupCompare-wrapped layers
            if b is not None: self.handles.append(m.register_forward_hook(lambda mod, i, o, b=b: o.clamp(-b, b)))
        self.handles.append(self.model.register_forward_hook(lambda mod, i, o: o.clamp(-self.logit_clip, self.logit_clip)))
        return self
    def __exit__(self, *a):
        for h in self.handles: h.remove()

# ----------------------------------------------------------------------------- selective reinforcement
class DupCompare(nn.Module):
    """Duplicate-and-compare wrapper: the wrapped layer is executed twice; elements on which the two executions disagree
    are recomputed by a third execution. Because weight faults are identical in all executions, only transient
    activation faults are detected, which is the intended behaviour of duplicate-and-compare at the layer level."""
    def __init__(self, inner: nn.Module):
        super().__init__(); self.inner = inner; self.disagreements = 0
    def forward(self, x):
        o1 = self.inner(x); o2 = self.inner(x)
        bad = o1 != o2
        if bad.any():
            self.disagreements += int(bad.sum()); o3 = self.inner(x); return torch.where(bad, o3, o1)
        return o1

def _set_module(model: nn.Module, name: str, new: nn.Module):
    parts = name.split("."); parent = model
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], new)

def apply_selective_redundancy(model: nn.Module, critical_blocks: Sequence[str]) -> List[str]:
    """Wrap every conv/linear layer belonging to a critical block (names as in Section III-G). Returns wrapped layer names.
    The wrapped inner module keeps its original name suffix '.inner', so injection hooks and weight flips still apply."""
    wrapped = []
    for n, m in list(injectable_layers(model).items()):
        if block_of(n) in critical_blocks and not isinstance(m, DupCompare):
            _set_module(model, n, DupCompare(m)); wrapped.append(n)
    return wrapped

def remove_selective_redundancy(model: nn.Module):
    for n, m in list(model.named_modules()):
        if isinstance(m, DupCompare): _set_module(model, n, m.inner)
