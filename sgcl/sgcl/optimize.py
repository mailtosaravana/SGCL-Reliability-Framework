"""Sensitivity-guided optimisation (Section III-I): sensitivity-aware pruning and mixed-precision allocation,
plus the analytical cost model of Section III-K."""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import copy, torch, torch.nn as nn, torch.nn.functional as F
from .common import DEVICE, Settings, injectable_layers, block_of, train, layer_macs, cost_change

# ----------------------------------------------------------------------------- pruning
def layer_prune_ratios(model, S: Dict[str, float], Settings_: Settings, scale: float = 1.0) -> Dict[str, float]:
    """r_k = 0.45 (S_k < 1.5 pp), 0.25 (1.5 <= S_k < T), 0 (S_k >= T), multiplied by `scale` for the sparsity sweep.
    The stem convolution and the classifier are never pruned."""
    r = {}
    for n in injectable_layers(model):
        b = block_of(n); s = S.get(b, 0.0)
        if b in ("conv1", "fc", "classifier", "features.0"): r[n] = 0.0; continue
        if s < Settings_.T_low_pp: r[n] = Settings_.prune_ratios[0]
        elif s < Settings_.T_pp: r[n] = Settings_.prune_ratios[1]
        else: r[n] = Settings_.prune_ratios[2]
        r[n] *= scale
    return r

@torch.no_grad()
def magnitude_masks(model, ratios: Dict[str, float], existing: Optional[Dict[str, torch.Tensor]] = None, criterion_weight: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    """Unstructured magnitude pruning per layer to the requested ratio (cumulative with existing masks).
    criterion_weight (RAP comparator): per-layer multiplier applied to |w| so that fault-sensitive layers keep more weights."""
    masks = {}
    for n, m in injectable_layers(model).items():
        w = m.weight.data.abs().clone()
        if criterion_weight: w = w * criterion_weight.get(n, 1.0)
        prev = existing[n] if existing and n in existing else torch.ones_like(w)
        r = ratios.get(n, 0.0)
        if r <= 0: masks[n] = prev; continue
        k = int(round(r * w.numel()))
        thresh = torch.kthvalue(w.flatten(), max(k, 1)).values
        masks[n] = ((w > thresh) & (prev > 0)).float()
        m.weight.data.mul_(masks[n])
    return masks

def global_sparsity(model, masks) -> float:
    tot = kept = 0
    for n, m in injectable_layers(model).items():
        tot += m.weight.numel(); kept += int(masks[n].sum()) if n in masks else m.weight.numel()
    return 1 - kept / tot

def sensitivity_aware_pruning(model, S, S_, train_dl, val_dl, target_scale: float = 1.0, steps: Optional[int] = None,
                              fat_lam_max: float = 0.0, criterion_weight=None, tag="prune") -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict]:
    """Iterative pruning in `steps` steps towards the layer-wise ratios of Table 2 (scaled by target_scale), with
    fine-tuning (ft_epochs, ft_lr, cosine) after each step. fat_lam_max > 0 fine-tunes under fault injection (RAP)."""
    steps = steps or S_.prune_steps; masks = None; log = []
    final = layer_prune_ratios(model, S, S_, target_scale)
    for s in range(1, steps + 1):
        frac = s / steps
        ratios = {n: 1 - (1 - r) ** frac for n, r in final.items()}   # cumulative ratio reaching r at the last step
        masks = magnitude_masks(model, ratios, masks, criterion_weight)
        sp = global_sparsity(model, masks)
        h = train(model, train_dl, val_dl, S_, epochs=S_.ft_epochs, lr=S_.ft_lr, prune_masks=masks, fat_lam_max=fat_lam_max, fat_curriculum=False, tag=f"{tag} step{s} sp={sp:.2f}")
        log.append({"step": s, "global_sparsity": sp, "val_acc": h["best_val"] if h["best_val"] else h["history"][-1]["val_acc"]})
    return model, masks, {"steps": log, "final_sparsity": global_sparsity(model, masks)}

# ----------------------------------------------------------------------------- mixed precision
class LowPrecision(nn.Module):
    """Stores the weight of a conv/linear layer in FP16 or FP8 (E4M3, per-channel symmetric scale) and dequantises
    at forward time. Faults are injected into the stored low-precision words (16 or 8 bit positions) and into the
    output activation quantised to the same format, so that the reduced bit width is faithfully represented."""
    def __init__(self, inner: nn.Module, fmt: str, calib_scale: Optional[torch.Tensor] = None):
        super().__init__(); object.__setattr__(self, "inner_cfg", inner); self.fmt = fmt   # not registered as a sub-module
        w = inner.weight.data
        if fmt == "fp16":
            self.scale = torch.ones(1, device=w.device); self.weight = nn.Parameter(w.half(), requires_grad=False)
        else:  # fp8 e4m3, per-output-channel scale
            s = (w.abs().flatten(1).max(1).values / 448.0).clamp_min(1e-8).view(-1, *([1] * (w.dim() - 1)))
            self.scale = s; self.weight = nn.Parameter((w / s).to(torch.float8_e4m3fn), requires_grad=False)
        self.bias = inner.bias
        # quantisation of the output activation: per-tensor symmetric scale from calibration (fp8) or plain cast (fp16)
        self.act_scale = calib_scale if calib_scale is not None else torch.ones(1, device=w.device)
    def deq_weight(self): return self.weight.data.to(torch.float32) * self.scale
    def forward(self, x):
        w = self.deq_weight(); c = self.inner_cfg
        if isinstance(c, nn.Conv2d): out = F.conv2d(x, w, self.bias, c.stride, c.padding, c.dilation, c.groups)
        else: out = F.linear(x, w, self.bias)
        return out
    def quantize_act(self, out):
        if self.fmt == "fp16": return out.half()
        return (out / self.act_scale).to(torch.float8_e4m3fn)
    def dequantize_act(self, q):
        if self.fmt == "fp16": return q.float()
        return q.to(torch.float32) * self.act_scale

@torch.no_grad()
def calibrate_act_scales(model, calib_dl, n_images: int) -> Dict[str, torch.Tensor]:
    scales = {}; hs = []
    for n, m in injectable_layers(model).items():
        def hook(mod, i, o, name=n): scales[name] = torch.maximum(scales.get(name, torch.zeros(1, device=o.device)), o.detach().abs().max().view(1))
        hs.append(m.register_forward_hook(hook))
    seen = 0
    for x, _ in calib_dl:
        model(x.to(DEVICE)); seen += x.shape[0]
        if seen >= n_images: break
    for h in hs: h.remove()
    return {n: (s / 448.0).clamp_min(1e-8) for n, s in scales.items()}

def _set_module(model, name, new):
    parts = name.split("."); parent = model
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], new)

MP_POLICIES = {  # Table 6
    "MP-1": "uniform FP16", "MP-2": "uniform FP8", "MP-3": "FP32 critical + FP16 others",
    "MP-4": "FP16 critical + FP8 others", "MP-5": "FP32 stem+head + FP8 others"}

def precision_policy(model, policy: str, critical_blocks: Sequence[str]) -> Dict[str, str]:
    prec = {}
    for n in injectable_layers(model):
        b = block_of(n); crit = b in critical_blocks; stemhead = b in ("conv1", "fc", "classifier", "features.0")
        prec[n] = {"MP-1": "fp16", "MP-2": "fp8", "MP-3": "fp32" if crit else "fp16", "MP-4": "fp16" if crit else "fp8",
                   "MP-5": "fp32" if stemhead else "fp8", "FP32": "fp32"}[policy]
    return prec

def apply_mixed_precision(model: nn.Module, prec: Dict[str, str], act_scales: Dict[str, torch.Tensor]) -> nn.Module:
    """Returns a deep copy of the model with LowPrecision wrappers where prec[n] != fp32."""
    m2 = copy.deepcopy(model)
    for n, m in list(injectable_layers(m2).items()):
        if prec.get(n, "fp32") != "fp32": _set_module(m2, n, LowPrecision(m, prec[n], act_scales.get(n)))
    return m2

def cost_of(model_fp32, prec: Dict[str, str], masks: Optional[Dict[str, torch.Tensor]] = None, duplicated: Sequence[str] = ()) -> Dict[str, float]:
    macs = layer_macs(model_fp32)
    keep = {n: float(masks[n].mean()) for n in masks} if masks else {}
    return cost_change(macs, prec, keep, duplicated)
