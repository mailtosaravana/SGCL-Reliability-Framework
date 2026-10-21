"""Statistical validation protocol (Section III-J): two-sided paired z-test on per-seed CER differences, 95 % CI,
Bonferroni correction with m = 4; adjusted p-values are compared against alpha = 0.05."""
from __future__ import annotations
import math
from typing import Dict, Sequence
import numpy as np
from scipy.stats import norm

def paired_z(cer_a: Sequence[float], cer_b: Sequence[float], m: int = 4) -> Dict[str, float]:
    """Difference d_i = CER_A,i - CER_B,i (A = reference/baseline, B = configuration). Positive mean = B is better."""
    a, b = np.asarray(cer_a, float), np.asarray(cer_b, float)
    assert a.shape == b.shape, "paired test needs the same seeds in both configurations"
    d = a - b; n = len(d); mean = d.mean(); sd = d.std(ddof=1) if n > 1 else 0.0; se = sd / math.sqrt(n) if n > 1 else float("nan")
    z = mean / se if se and se > 0 else float("nan")
    p = 2 * (1 - norm.cdf(abs(z))) if not math.isnan(z) else float("nan")
    return {"n": n, "delta_cer_pp": 100 * mean, "se_pp": 100 * se, "ci95_pp": [100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se)],
            "z": z, "p_raw": p, "p_adj": min(1.0, m * p) if not math.isnan(p) else float("nan"), "m": m,
            "significant_adj_0p05": bool(not math.isnan(p) and m * p < 0.05),
            "rif": float(a.mean() / b.mean()) if b.mean() > 0 else float("inf")}

def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)): return "–"
    if p == 0: return "<1e-300"
    e = int(math.floor(math.log10(p))); return f"{p/10**e:.1f}×10^{e}" if p < 1e-3 else f"{p:.3f}"
