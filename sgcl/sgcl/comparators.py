"""Matched re-implementations of the four single-layer comparators of Table 10, each evaluated under the same
dataset split, architecture, fault model, seeds and lambda as the proposed framework (Section IV-H).

  range_restriction  [34]  activation-range restriction only (calibrated per-layer bounds + logit clip)
  fat_rabbit         [35]  fault-aware training against bit flips with a sharpness-aware optimiser (SAM), fixed lambda
  rap                [40]  reliability-aware pruning: pruning criterion weighted by layer fault sensitivity, fine-tuning
                           under fault injection, same 30 % global sparsity as the proposed pruning
  rl_bit_protection  [36]  reinforcement-learning selection of (block, bit-field) pairs to protect under a budget,
                           protection modelled as detection-and-restore of flips on the protected bits

These are faithful re-implementations of the published mechanisms at the granularity of this study; they are not
the authors' original code."""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import copy, math, numpy as np, torch
from .common import DEVICE, Settings, injectable_layers, block_of, train, monte_carlo, accuracy, FaultInjector
from .optimize import sensitivity_aware_pruning, layer_prune_ratios

# ----------------------------------------------------------------------------- FAT-RABBIT
def train_fat_rabbit(model, train_dl, val_dl, S: Settings, lam: float, ckpt: str, rho: float = 0.05):
    """Bit-flip-aware training at a constant lambda (no curriculum) with SAM (rho = 0.05)."""
    return train_fixed_lambda(model, train_dl, val_dl, S, lam, ckpt, sam_rho=rho, tag="fat-rabbit")

def train_fixed_lambda(model, train_dl, val_dl, S, lam, ckpt, sam_rho=0.0, tag=""):
    return train(model, train_dl, val_dl, S, fat_lam_max=lam, fat_curriculum=False, sam_rho=sam_rho, ckpt=ckpt, tag=tag)

# ----------------------------------------------------------------------------- RAP
def rap_pruning(model, S_scores: Dict[str, float], S: Settings, train_dl, val_dl, lam: float, target_scale: float = 1.0):
    """Reliability-aware pruning: criterion |w| x (1 + S_k / max S) keeps weights of fault-sensitive layers, fine-tuning
    under fault injection at lambda; layer-wise ratios chosen to match the global sparsity of the proposed pruning."""
    mx = max(max(S_scores.values()), 1e-9) if S_scores else 1.0
    crit = {n: 1.0 + max(S_scores.get(block_of(n), 0.0), 0.0) / mx for n in injectable_layers(model)}   # S_k clamped at 0 (noise)
    return sensitivity_aware_pruning(model, S_scores, S, train_dl, val_dl, target_scale, fat_lam_max=lam, criterion_weight=crit, tag="rap")

# ----------------------------------------------------------------------------- RL bit protection
FIELDS = {"sign": (31, 31), "exponent": (23, 30), "mantissa": (0, 22)}

def _protected_masks(model, chosen: Sequence[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
    """Element-level protection masks: a chosen (block, field) protects the given bit field of every weight in the block.
    Since flip_bits_ draws one bit per element, protection is approximated at element granularity by suppressing a flip
    when its bit lies in a protected field: implemented by storing, per layer, the protected field set."""
    masks = {}
    for n, m in injectable_layers(model).items():
        fields = [f for (b, f) in chosen if b == block_of(n)]
        if fields: masks[n] = fields
    return masks

class FieldProtectedInjector(FaultInjector):
    """FaultInjector variant in which flips whose bit position falls in a protected field of a protected layer are
    suppressed (detection-and-restore)."""
    def __init__(self, model, lam, seed, dist, protected_fields: Dict[str, List[str]]):
        super().__init__(model, lam, seed, dist); self.pf = protected_fields
    def __enter__(self):
        import sgcl.common as C
        for n, m in self.layers.items():
            self.backup[n] = m.weight.data.clone()
            self.n_flips += self._flip(m.weight.data, n)
        for n, m in self.layers.items(): self.handles.append(m.register_forward_hook(lambda mod, i, o, name=n: self._act(mod, o, name), prepend=True))
        return self
    def _act(self, module, out, name):
        with torch.no_grad(): f = out.detach().clone(); self.n_flips += self._flip(f, name)
        return f
    def _flip(self, t, name):
        import sgcl.common as C
        if self.lam <= 0: return 0
        n = t.numel(); k = int(torch.binomial(torch.tensor(float(n), device=t.device), torch.tensor(self.lam, device=t.device), generator=self.gen).item())
        if k == 0: return 0
        idx = torch.unique(torch.randint(0, n, (k,), generator=self.gen, device=t.device))
        bits = C.sample_bit_positions(idx.numel(), self.gen, t.device, self.dist, 32)
        keep = torch.ones_like(bits, dtype=torch.bool)
        for f in self.pf.get(name, []):
            lo, hi = FIELDS[f]; keep &= ~((bits >= lo) & (bits <= hi))
        idx, bits = idx[keep], bits[keep]
        if idx.numel() == 0: return 0
        flat = t.reshape(-1); iv = flat[idx].view(torch.int32)
        flat[idx] = (iv ^ (torch.ones_like(iv) << bits.to(torch.int32))).view(t.dtype)
        return int(idx.numel())

def rl_bit_protection(model, blocks: Sequence[str], val_dl, S: Settings, lam: float, clean_acc: float, budget: int = 4,
                      episodes: int = 40, eval_trials: int = 8, seed: int = 0) -> Tuple[List[Tuple[str, str]], Dict]:
    """Tabular Q-learning over actions (block, field). An episode selects `budget` actions greedily (epsilon-greedy),
    the reward is -CER on the validation subset estimated from `eval_trials` paired seeds, and Q-values are updated by
    incremental averaging. Returns the greedy protection set and the learning log."""
    actions = [(b, f) for b in blocks for f in FIELDS]
    Q = {a: 0.0 for a in actions}; N = {a: 0 for a in actions}; rng = np.random.default_rng(seed); log = []
    def reward(chosen):
        pf = _protected_masks(model, chosen); accs = []
        for i in range(eval_trials):
            with FieldProtectedInjector(model, lam, 100_000 + i, "uniform", pf): accs.append(accuracy(model, val_dl))
        return -(clean_acc - float(np.mean(accs)))
    for ep in range(episodes):
        eps = max(0.05, 1 - ep / episodes); chosen = []
        for _ in range(budget):
            avail = [a for a in actions if a not in chosen]
            a = avail[rng.integers(len(avail))] if rng.random() < eps else max(avail, key=lambda a: Q[a])
            chosen.append(a)
        r = reward(chosen)
        for a in chosen: N[a] += 1; Q[a] += (r - Q[a]) / N[a]
        log.append({"episode": ep, "reward": r, "chosen": chosen}); print(f"   RL ep {ep} reward {r*100:.3f} pp {chosen}", flush=True)
    best = sorted(actions, key=lambda a: -Q[a])[:budget]
    return best, {"Q": {f"{b}/{f}": q for (b, f), q in Q.items()}, "log": log, "budget": budget}

def protection_overhead(model, chosen) -> Dict[str, float]:
    """Analytical overhead of detection-and-restore: one parity bit per protected field per weight (memory) and a
    compare per protected weight (latency ~ MAC fraction of protected layers x 0.05, energy likewise)."""
    tot = prot = 0
    for n, m in injectable_layers(model).items():
        tot += m.weight.numel()
        if any(b == block_of(n) for b, _ in chosen): prot += m.weight.numel()
    frac = prot / max(tot, 1)
    return {"latency_pct": 100 * frac * 0.05, "memory_pct": 100 * frac * (len(set(f for _, f in chosen)) / 32), "energy_pct": 100 * frac * 0.05}

def monte_carlo_protected(model, dl, lam, n_trials, clean_acc, chosen, dist="uniform", log_every=50, desc="rl"):
    pf = _protected_masks(model, chosen); accs = []
    for i in range(n_trials):
        with FieldProtectedInjector(model, lam, i, dist, pf): accs.append(accuracy(model, dl))
        if log_every and (i + 1) % log_every == 0: print(f"   {desc} trial {i+1}/{n_trials}", flush=True)
    accs = np.array(accs); cer = clean_acc - accs
    return {"lambda": lam, "n_trials": n_trials, "clean_acc": clean_acc, "acc_per_seed": accs.tolist(), "cer_per_seed": cer.tolist(),
            "acc_mean": float(accs.mean()), "acc_var": float(accs.var(ddof=1)) if n_trials > 1 else 0.0, "cer_mean": float(cer.mean()),
            "cer_se": float(cer.std(ddof=1) / math.sqrt(n_trials)) if n_trials > 1 else 0.0}
