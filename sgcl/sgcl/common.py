"""
Sensitivity-Guided Cross-Layer (SGCL) reliability framework -- shared components.

Data, models, training loop, IEEE-754 bit-flip fault injection (Section III-C / Algorithm 1) and
utilities. Everything is seeded; every random draw used for fault injection comes from a torch.Generator
seeded with the trial index, so that trial i injects the same faults into every configuration whose
injectable tensors have the same shapes (paired design, Section III-J).
"""
from __future__ import annotations
import os, json, math, time, random, hashlib, dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------- configuration
@dataclass
class Settings:
    """All numerical settings of Table 2. `budget="paper"` reproduces the protocol of the manuscript;
    `budget="colab"` is a reduced protocol that fits a free Colab/Kaggle GPU session (same structure,
    fewer epochs and a stratified evaluation subset per trial). The budget used is recorded in every output."""
    budget: str = "colab"
    data_root: str = "./data"
    out_root: str = "./results"
    epochs: int = 200
    ft_epochs: int = 30            # fine-tuning per pruning step
    batch: int = 128
    lr: float = 0.1
    ft_lr: float = 0.01
    momentum: float = 0.9
    wd: float = 5e-4
    n_trials: int = 500            # Monte Carlo seeds 0..N-1
    n_sens_trials: int = 100       # trials per layer for sensitivity estimation
    eval_subset: int = 0           # 0 = full test split; >0 = stratified subset size per trial
    train_subset: int = 0          # 0 = full training split (smoke budget only uses a subset)
    lambdas: Tuple[float, ...] = (1e-6, 1e-5, 1e-4)
    lam_op: float = 1e-5           # operating point
    lam_fat_max: float = 2e-5      # curriculum FAT maximum
    T_pp: float = 2.5              # sensitivity threshold (percentage points)
    T_low_pp: float = 1.5
    prune_ratios: Tuple[float, float, float] = (0.45, 0.25, 0.0)   # S<1.5 | 1.5<=S<T | S>=T
    prune_steps: int = 3
    clip_percentile: float = 99.99
    clip_factor: float = 1.1
    logit_clip: float = 20.0
    beta_res: float = 1e-4
    calib_images: int = 5000
    mp_calib_images: int = 512
    train_seed: int = 0
    workers: int = 2
    fat_weights: bool = True       # inject weight faults (in addition to activation faults) during fault-aware training

    @staticmethod
    def for_budget(b: str, **kw) -> "Settings":
        s = Settings(budget=b)
        if b == "colab":
            s.epochs, s.ft_epochs, s.n_trials, s.n_sens_trials, s.eval_subset = 30, 5, 500, 60, 2000
        elif b == "fast":     # one free Colab day: N = 500 kept, smaller evaluation subset and shorter training
            s.epochs, s.ft_epochs, s.n_trials, s.n_sens_trials, s.eval_subset = 20, 3, 500, 40, 1000
        elif b == "smoke":
            s.epochs, s.ft_epochs, s.n_trials, s.n_sens_trials, s.eval_subset, s.calib_images, s.mp_calib_images, s.train_subset = 1, 1, 4, 2, 128, 128, 64, 512
        for k, v in kw.items(): setattr(s, k, v)
        return s

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=1)

# ----------------------------------------------------------------------------- data
MEANS = {"cifar10": (0.4914, 0.4822, 0.4465), "gtsrb": (0.3403, 0.3121, 0.3214)}
STDS = {"cifar10": (0.2470, 0.2435, 0.2616), "gtsrb": (0.2724, 0.2608, 0.2669)}
NUM_CLASSES = {"cifar10": 10, "gtsrb": 43}

def get_datasets(name: str, root: str, seed: int = 0):
    """CIFAR-10: 45,000 / 5,000 / 10,000.  GTSRB: 35,289 / 3,920 / 12,630, all 32x32 (Table 2)."""
    norm = T.Normalize(MEANS[name], STDS[name])
    if name == "cifar10":
        tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
        te_tf = T.Compose([T.ToTensor(), norm])
        imgdir = os.path.join(root, "CIFAR-10-images")        # optional image-folder mirror (train/<class>/*.jpg, test/<class>/*.jpg)
        if os.path.isdir(imgdir):
            IF = torchvision.datasets.ImageFolder
            full, full_eval, test = IF(f"{imgdir}/train", tr_tf), IF(f"{imgdir}/train", te_tf), IF(f"{imgdir}/test", te_tf)
        else:
            full = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr_tf)
            full_eval = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=te_tf)
            test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te_tf)
        n_val = 5000
    else:
        tr_tf = T.Compose([T.Resize((32, 32)), T.RandomAffine(5, translate=(0.05, 0.05)), T.ToTensor(), norm])
        te_tf = T.Compose([T.Resize((32, 32)), T.ToTensor(), norm])
        full = torchvision.datasets.GTSRB(root, split="train", download=True, transform=tr_tf)
        full_eval = torchvision.datasets.GTSRB(root, split="train", download=True, transform=te_tf)
        test = torchvision.datasets.GTSRB(root, split="test", download=True, transform=te_tf)
        n_val = 3920
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full), generator=g).tolist()
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    train = torch.utils.data.Subset(full, tr_idx)
    train_eval = torch.utils.data.Subset(full_eval, tr_idx)      # train images without augmentation (calibration)
    val = torch.utils.data.Subset(full_eval, va_idx)
    return train, train_eval, val, test

def stratified_subset(dataset, n: int, num_classes: int, seed: int = 0):
    """Fixed stratified subset of the test split used for every trial when eval_subset > 0 (colab budget)."""
    if n <= 0 or n >= len(dataset): return dataset
    if hasattr(dataset, "targets"): targets = np.array([int(t) for t in dataset.targets])
    elif hasattr(dataset, "_samples"): targets = np.array([s[1] for s in dataset._samples])
    else: targets = np.array([int(dataset[i][1]) for i in range(len(dataset))])
    rng = np.random.default_rng(seed); idx = []
    per = n // num_classes
    for c in range(num_classes):
        ci = np.where(targets == c)[0]; idx += rng.choice(ci, min(per, len(ci)), replace=False).tolist()
    return torch.utils.data.Subset(dataset, sorted(idx))

def loader(ds, batch, shuffle=False, workers=2):
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())

def preload(ds, batch) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Materialise a (small) evaluation set as device-resident batches so that the N Monte Carlo trials do not pay
    the data-loading cost N times. The images are identical in every trial (paired design)."""
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    return [(x.to(DEVICE), y.to(DEVICE)) for x, y in dl]

# ----------------------------------------------------------------------------- models (32x32 variants)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(out)
        self.downsample = None
        if stride != 1 or inp != out:
            self.downsample = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))
        self.res_out = None          # residual-branch output (for the regularisation term beta, Section III-G)
    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        r = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        self.res_out = r
        return F.relu(r + idt)

class ResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs: 3x3 stem, no max-pool, stages layer1..layer4 with two BasicBlocks each.
    Layer names follow torchvision (conv1, layer1.0 ... layer4.1, fc) as used in the manuscript."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
    def residual_blocks(self): return [b for l in (self.layer3, self.layer4) for b in l]

def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18": return ResNet18(num_classes)
    if arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=None, num_classes=num_classes)
        m.features[0][0].stride = (1, 1)                      # 32x32 inputs: stem stride 1
        return m
    if arch == "vgg16":
        m = torchvision.models.vgg16_bn(weights=None, num_classes=num_classes)
        m.avgpool = nn.AdaptiveAvgPool2d(1); m.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512, num_classes))
        return m
    if arch == "mobilenet_v2":
        m = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
        m.features[0][0].stride = (1, 1); return m
    raise ValueError(arch)

def injectable_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """Weights and output activations of every convolutional and linear layer (Section III-C)."""
    return {n: m for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear)) or type(m).__name__ == "LowPrecision"}

def block_of(layer_name: str) -> str:
    """Map a module name to the block granularity used for sensitivity scores (conv1, layer1.0, ..., fc)."""
    parts = layer_name.split(".")
    if parts[0].startswith("layer") and len(parts) >= 2: return parts[0] + "." + parts[1]
    return parts[0] if parts[0] in ("conv1", "fc", "classifier") else ".".join(parts[:2])

# ----------------------------------------------------------------------------- bit-flip fault injection
BIT_DISTS = {  # probability mass over (sign, exponent, mantissa) fields, Section IV-I
    "uniform": (1 / 32, 8 / 32, 23 / 32), "exponent": (0.05, 0.60, 0.35), "mantissa": (0.05, 0.10, 0.85)}

def sample_bit_positions(n: int, gen: torch.Generator, device, dist: str = "uniform", width: int = 32) -> torch.Tensor:
    """Bit positions for n flips. For 32-bit floats: sign=31, exponent=30..23, mantissa=22..0.
    For 16-bit floats (mixed precision): sign=15, exponent=14..10, mantissa=9..0; for FP8 E4M3: 7, 6..3, 2..0."""
    if width == 32: fields = [(31, 31), (23, 30), (0, 22)]
    elif width == 16: fields = [(15, 15), (10, 14), (0, 9)]
    else: fields = [(7, 7), (3, 6), (0, 2)]
    p = torch.tensor(BIT_DISTS[dist], device=device)
    if dist == "uniform":  # exactly uniform over all positions
        return torch.randint(0, width, (n,), generator=gen, device=device)
    fld = torch.multinomial(p, n, replacement=True, generator=gen)
    lo = torch.tensor([f[0] for f in fields], device=device)[fld]; hi = torch.tensor([f[1] for f in fields], device=device)[fld]
    return lo + (torch.rand(n, generator=gen, device=device) * (hi - lo + 1).float()).long().clamp(max=hi - lo)

def flip_bits_(t: torch.Tensor, lam: float, gen: torch.Generator, dist: str = "uniform", protected_mask: Optional[torch.Tensor] = None) -> int:
    """In-place per-element Bernoulli(lam) single-bit flips in the IEEE-754 representation of t (Algorithm 1, lines 2-4).
    Returns the number of flipped elements. protected_mask (same shape, bool) marks elements whose bits are protected
    (RL bit protection comparator): flips on protected elements are suppressed."""
    if lam <= 0: return 0
    n = t.numel()
    k = int(torch.binomial(torch.tensor(float(n), device=t.device), torch.tensor(lam, device=t.device), generator=gen).item())
    if k == 0: return 0
    idx = torch.unique(torch.randint(0, n, (k,), generator=gen, device=t.device))   # distinct elements (one flip each)
    if t.dtype == torch.float32: width, itype = 32, torch.int32
    elif t.dtype in (torch.float16, torch.bfloat16): width, itype = 16, torch.int16
    else: width, itype = 8, torch.int8
    bits = sample_bit_positions(idx.numel(), gen, t.device, dist, width)
    if protected_mask is not None:
        keep = ~protected_mask.reshape(-1)[idx]; idx, bits = idx[keep], bits[keep]
        if idx.numel() == 0: return 0
    flat = t.reshape(-1)
    iv = flat[idx].view(itype)
    masks = (torch.ones_like(iv) << bits.to(itype))
    flat[idx] = (iv ^ masks).view(t.dtype)
    return int(idx.numel())

class FaultInjector:
    """Activation + weight injector for one Monte Carlo trial. Weight flips are applied in place and restored
    afterwards (so the clean checkpoint is reused); activation flips are applied by forward hooks on the output of
    every conv/linear layer (optionally restricted to a set of layer names for the sensitivity analysis)."""
    def __init__(self, model: nn.Module, lam: float, seed: int, dist: str = "uniform", layers: Optional[Sequence[str]] = None,
                 weights: bool = True, activations: bool = True, protected: Optional[Dict[str, torch.Tensor]] = None, sanitize: bool = False):
        """sanitize=True (fault-aware training only): corrupted activations that are non-finite or exceed FAT_ACT_CAP x the
        largest clean activation of the tensor are replaced by 0 (NaN/overflow filtering), so that the training forward and
        backward passes stay finite. Evaluation never sanitises."""
        self.model, self.lam, self.dist, self.sanitize, self.n_sanitized = model, lam, dist, sanitize, 0
        self.gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
        self.layers = {n: m for n, m in injectable_layers(model).items() if layers is None or n in layers}
        self.weights, self.activations, self.protected = weights, activations, protected or {}
        self.handles, self.backup, self.n_flips = [], {}, 0
    def __enter__(self):
        if self.weights:
            for n, m in self.layers.items():
                self.backup[n] = m.weight.data.clone()
                self.n_flips += flip_bits_(m.weight.data, self.lam, self.gen, self.dist, self.protected.get(n))
        if self.activations:
            for n, m in self.layers.items():
                self.handles.append(m.register_forward_hook(self._hook, prepend=True))   # before clipping hooks
        return self
    def _hook(self, module, inp, out):
        with torch.no_grad():
            if hasattr(module, "quantize_act"):      # low-precision layer: flip bits in the reduced-width representation
                q = module.quantize_act(out.detach()).clone(); self.n_flips += flip_bits_(q, self.lam, self.gen, self.dist); f = module.dequantize_act(q)
            else:
                f = out.detach().clone(); self.n_flips += flip_bits_(f, self.lam, self.gen, self.dist)
            if self.sanitize:
                a = out.detach().abs().flatten(); ref = a[torch.randint(0, a.numel(), (min(a.numel(), 65_536),), device=a.device)].median()
                bad = ~torch.isfinite(f) | (f.abs() > FAT_ACT_CAP * ref.clamp_min(1e-3))
                self.n_sanitized += int(bad.sum()); f = torch.where(bad, torch.zeros_like(f), f)
                keep = (~bad).to(out.dtype)
        if not out.requires_grad: return f
        if self.sanitize: return out * keep + (f - out.detach() * keep)   # straight-through, no gradient through filtered elements
        return out + (f - out.detach())                                   # straight-through: corrupted value, intact gradient (FAT)
    def __exit__(self, *a):
        for h in self.handles: h.remove()
        for n, m in self.layers.items():
            if n in self.backup: m.weight.data.copy_(self.backup[n])
        self.backup.clear()

# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def accuracy(model: nn.Module, dl, max_batches: Optional[int] = None) -> float:
    model.eval(); correct = n = 0
    for b, (x, y) in enumerate(dl):
        if max_batches and b >= max_batches: break
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item(); n += y.numel()
    return correct / max(n, 1)

def monte_carlo(model: nn.Module, dl, lam: float, n_trials: int, clean_acc: float, dist="uniform", layers=None,
                weights=True, activations=True, protected=None, log_every=50, desc="") -> Dict:
    """Algorithm 1: N paired trials with seeds 0..N-1. Returns per-seed accuracy and CER (Eq. 4)."""
    accs = []; t0 = time.time()
    for i in range(n_trials):
        with FaultInjector(model, lam, i, dist, layers, weights, activations, protected) as fi:
            a = accuracy(model, dl)
        accs.append(a)
        if log_every and (i + 1) % log_every == 0: print(f"   {desc} trial {i+1}/{n_trials}  acc={a*100:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    accs = np.array(accs); cer = clean_acc - accs
    return {"lambda": lam, "n_trials": n_trials, "clean_acc": clean_acc, "acc_per_seed": accs.tolist(), "cer_per_seed": cer.tolist(),
            "acc_mean": float(accs.mean()), "acc_var": float(accs.var(ddof=1)) if n_trials > 1 else 0.0, "cer_mean": float(cer.mean()),
            "cer_se": float(cer.std(ddof=1) / math.sqrt(n_trials)) if n_trials > 1 else 0.0}

# ----------------------------------------------------------------------------- training
FAT_ACT_CAP = 1e4      # training-time overflow filter (see FaultInjector.sanitize)
FAT_LOSS_CAP = 100.0   # a per-sample cross-entropy above 100 nats means the logits overflowed under an injected fault; such samples
                       # are excluded from the fault-aware loss together with NaN/Inf samples (DieHardNet-style NaN filtering)

def train(model: nn.Module, train_dl, val_dl, S: Settings, epochs: Optional[int] = None, lr: Optional[float] = None,
          fat_lam_max: float = 0.0, fat_dist="uniform", beta_res: float = 0.0, ckpt: Optional[str] = None,
          sam_rho: float = 0.0, prune_masks: Optional[Dict[str, torch.Tensor]] = None, tag="", fat_curriculum: bool = True) -> Dict:
    """SGD (momentum 0.9, cosine schedule, weight decay 5e-4, batch 128). Options implement:
       fat_lam_max > 0 : curriculum fault-aware training, lambda_e = lam_max * e / E, faults in the forward pass only (Alg. 2)
       beta_res > 0    : residual-path regularisation on layer3/layer4 (Section III-G)
       sam_rho > 0     : sharpness-aware minimisation step (FAT-RABBIT comparator)
       prune_masks     : keep pruned weights at zero (sensitivity-aware pruning fine-tuning / RAP)"""
    epochs = epochs or S.epochs; lr = lr or S.lr
    model.to(DEVICE); opt = torch.optim.SGD(model.parameters(), lr, momentum=S.momentum, weight_decay=S.wd, nesterov=False)
    steps = epochs * len(train_dl); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []; layers = injectable_layers(model); best = 0.0; t_start = time.time()
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    def apply_masks():
        if prune_masks:
            for n, m in layers.items():
                if n in prune_masks: m.weight.data.mul_(prune_masks[n])
    for ep in range(epochs):
        model.train(); lam_e = fat_lam_max * (ep / max(epochs - 1, 1) if fat_curriculum else 1.0)   # Eq. (8): 0 at the first epoch, lam_max at the last
        tot = 0.0; nb = 0; t0 = time.time(); dropped = [0]; seen = [0]; sanitized = [0]; skipped = [0]
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            def loss_fn(faulty: bool = False):
                out = model(x)
                ls = F.cross_entropy(out, y, reduction="none")                       # per-sample loss
                if beta_res > 0 and hasattr(model, "residual_blocks"):
                    ls = ls + beta_res * sum(b.res_out.pow(2).flatten(1).mean(1) for b in model.residual_blocks() if b.res_out is not None)
                if not faulty: return ls.mean()
                finite = torch.isfinite(ls) & (ls < FAT_LOSS_CAP)                     # NaN/Inf filtering (faulty pass only): samples
                dropped[0] += int((~finite).sum())                                    # whose forward pass overflowed are excluded
                return ls[finite].mean() if finite.any() else ls.sum() * 0.0
            def step_grads():
                """Algorithm 2, one step: accumulates the gradients of the step into .grad and returns the (detached) loss.
                lam_e = 0: ordinary loss. lam_e > 0: clean pass (updates the batch-norm statistics) plus a faulty pass with
                the bit-flip injector, back-propagated while the faults are still applied; in the faulty pass batch-norm uses
                its running statistics so that an overflowed activation of one image cannot corrupt the whole batch, and
                overflowed samples / activations are excluded (NaN filtering). The two losses are averaged."""
                if lam_e <= 0:
                    l = loss_fn(); l.backward(); return l.detach()
                l_clean = loss_fn(); (0.5 * l_clean).backward(); seen[0] += y.numel()
                for bn in bn_layers: bn.eval()
                with FaultInjector(model, lam_e, S.train_seed * 10_000_000 + ep * 100_000 + bi, fat_dist, weights=S.fat_weights, sanitize=True) as fi:
                    l_fault = loss_fn(faulty=True); (0.5 * l_fault).backward()          # gradients w.r.t. the corrupted forward pass
                sanitized[0] += fi.n_sanitized
                for bn in bn_layers: bn.train()
                return (0.5 * (l_clean + l_fault)).detach()
            opt.zero_grad(set_to_none=True); loss = step_grads()
            if lam_e > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # bounded update under corrupted activations
            if sam_rho > 0:   # SAM: perturb to the worst-case neighbourhood, recompute gradient, restore
                with torch.no_grad():
                    gn = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])) + 1e-12
                    eps = [(p, sam_rho * p.grad / gn) for p in model.parameters() if p.grad is not None]
                    for p, e in eps: p.add_(e)
                opt.zero_grad(set_to_none=True); step_grads()
                with torch.no_grad():
                    for p, e in eps: p.sub_(e)
            if torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): opt.step()
            else: skipped[0] += 1
            sched.step(); apply_masks()
            tot += float(loss.item()) if torch.isfinite(loss) else 0.0; nb += 1
        va = accuracy(model, val_dl)
        hist.append({"epoch": ep + 1, "loss": tot / nb, "val_acc": va, "lambda_e": lam_e, "dropped_samples_frac": dropped[0] / max(seen[0], 1), "sanitized_elements": sanitized[0], "skipped_steps": skipped[0], "time_s": time.time() - t0})
        print(f"[{tag}] epoch {ep+1}/{epochs} loss {tot/nb:.3f} val {va*100:.2f} lam_e {lam_e:.1e} dropped {100*dropped[0]/max(seen[0],1):.2f}% sanitized {sanitized[0]} skipped {skipped[0]} ({time.time()-t0:.0f}s)", flush=True)
        if ckpt and va >= best:
            best = va; os.makedirs(os.path.dirname(ckpt), exist_ok=True); torch.save(model.state_dict(), ckpt)
    if ckpt and os.path.exists(ckpt): model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return {"history": hist, "best_val": best, "train_time_s": time.time() - t_start, "epochs": epochs}

# ----------------------------------------------------------------------------- analytical cost model (Section III-K)
COST = {"fp32": (1.0, 1.0, 1.0), "fp16": (0.5, 0.5, 0.6), "fp8": (0.3, 0.25, 0.4)}   # latency, memory, energy factors

@torch.no_grad()
def layer_macs(model: nn.Module, input_shape=(1, 3, 32, 32)) -> Dict[str, Tuple[int, int]]:
    """Per-layer (MACs, parameter count) obtained with forward hooks on a single input."""
    res = {}; hs = []
    for n, m in injectable_layers(model).items():
        def hook(mod, inp, out, name=n):
            if isinstance(mod, nn.Conv2d):
                macs = out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            else: macs = mod.in_features * mod.out_features
            res[name] = (int(macs), int(mod.weight.numel()))
        hs.append(m.register_forward_hook(hook))
    model.eval(); model(torch.zeros(input_shape, device=next(model.parameters()).device))
    for h in hs: h.remove()
    return res

def cost_change(macs: Dict[str, Tuple[int, int]], precision: Dict[str, str], keep_ratio: Dict[str, float], duplicated: Sequence[str] = ()) -> Dict[str, float]:
    """Relative latency / memory / energy change vs. uniform FP32 without pruning or duplication."""
    base = [sum(v[0] for v in macs.values()), sum(v[1] for v in macs.values()), sum(v[0] for v in macs.values())]
    lat = mem = en = 0.0
    for n, (mc, pc) in macs.items():
        f = COST[precision.get(n, "fp32")]; kr = keep_ratio.get(n, 1.0); dup = 2.0 if n in duplicated else 1.0
        lat += mc * kr * f[0] * dup; mem += pc * kr * f[1]; en += mc * kr * f[2] * dup
    return {"latency_pct": 100 * (lat / base[0] - 1), "memory_pct": 100 * (mem / base[1] - 1), "energy_pct": 100 * (en / base[2] - 1)}
