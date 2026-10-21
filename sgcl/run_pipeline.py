#!/usr/bin/env python3
"""
Sensitivity-Guided Cross-Layer (SGCL) framework -- end-to-end experimental pipeline.

Every stage writes its results to <out_root>/<arch>_<dataset>/*.json and is skipped on re-run if the file exists,
so an interrupted Colab session can simply be restarted. Checkpoints go to <out_root>/ckpt/.

    python run_pipeline.py --budget colab --out results --data data            # everything
    python run_pipeline.py --budget colab --stages baseline,vuln,sens          # subset of stages
    python run_pipeline.py --budget paper ...                                   # full protocol of the manuscript

Stages: baseline, vuln, sens, arch, fat, opt, unified, comparators, bitdist   (gtsrb runs through the same stages)
"""
from __future__ import annotations
import os, sys, json, time, argparse, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgcl.common import (Settings, DEVICE, seed_everything, save_json, get_datasets, stratified_subset, loader, preload, build_model,
                         accuracy, monte_carlo, train, injectable_layers, NUM_CLASSES, layer_macs, cost_change, block_of)
from sgcl.hardening import (layer_sensitivity, critical_set, calibrate_bounds, RangeRestriction, apply_selective_redundancy, remove_selective_redundancy)
from sgcl.optimize import (sensitivity_aware_pruning, calibrate_act_scales, precision_policy, apply_mixed_precision, cost_of, MP_POLICIES, global_sparsity)
from sgcl.comparators import train_fat_rabbit, rap_pruning, rl_bit_protection, monte_carlo_protected, protection_overhead
from sgcl.stats import paired_z

p = argparse.ArgumentParser()
p.add_argument("--budget", default="colab", choices=["paper", "colab", "fast", "smoke"])
p.add_argument("--out", default="results"); p.add_argument("--data", default="data")
p.add_argument("--stages", default="baseline,vuln,sens,arch,fat,opt,unified,comparators,bitdist")
p.add_argument("--pairs", default="resnet18:cifar10,efficientnet_b0:cifar10,resnet18:gtsrb,efficientnet_b0:gtsrb")
p.add_argument("--extra-archs", default="", help="comma list, e.g. vgg16,mobilenet_v2 (Table 5, CIFAR-10 FAT only)")
p.add_argument("--n-trials", type=int, default=None); p.add_argument("--epochs", type=int, default=None)
p.add_argument("--eval-subset", type=int, default=None); p.add_argument("--workers", type=int, default=2)
p.add_argument("--lambdas", default="1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4", help="lambda sweep for the baseline / FAT curves (Fig. 2, Fig. 5)")
p.add_argument("--lam-op", type=float, default=None, help="operating point lambda used by every other stage (default: Table 2 value 1e-5)")
p.add_argument("--lam-fat-max", type=float, default=None, help="curriculum FAT maximum (default: 2 x lam-op)")
p.add_argument("--lam-sens", type=float, default=None, help="lambda for the layer-wise sensitivity estimate (default: lam-op)")
p.add_argument("--critical-rule", default="median", choices=["threshold", "median"], help="critical set: fixed T (2.5 pp) or median of S_k")
p.add_argument("--fat-weights", type=int, default=1, help="1: inject weights and activations during FAT (as at evaluation); 0: activations only")
args = p.parse_args()
kw = {k: v for k, v in {"n_trials": args.n_trials, "epochs": args.epochs, "eval_subset": args.eval_subset, "workers": args.workers, "lam_op": args.lam_op}.items() if v is not None}
S = Settings.for_budget(args.budget, data_root=args.data, out_root=args.out, **kw)
S.lam_fat_max = args.lam_fat_max if args.lam_fat_max else 2 * S.lam_op
S.fat_weights = bool(args.fat_weights); S.lam_sens = args.lam_sens or S.lam_op; S.critical_rule = args.critical_rule
SWEEP = sorted(set(float(x) for x in args.lambdas.split(",")) | {S.lam_op})
PRIMARY = {S.lam_op, S.lam_op / 10, S.lam_op * 10}              # Table 3 rows: N trials; other sweep points: N2 trials
S.sweep = SWEEP
N2 = {"paper": S.n_trials, "smoke": S.n_trials, "fast": 100}.get(S.budget, max(50, S.n_trials // 2))       # trials for secondary sweeps (Fig. 5, Fig. 6 non-30 % points)
STAGES = args.stages.split(","); PAIRS = [tuple(x.split(":")) for x in args.pairs.split(",")]
FULL = {("resnet18", "cifar10"), ("resnet18", "gtsrb")}      # pairs that run the complete cross-layer pipeline
torch.backends.cudnn.benchmark = True
print(f"device {DEVICE}  budget {S.budget}  settings {json.dumps(S.__dict__)}", flush=True)
save_json(S.__dict__, f"{S.out_root}/settings.json")

OP = f"op{S.lam_op:.0e}"                                         # everything that depends on the operating point lives here
SHARED = lambda name: name == "train_baseline" or name.startswith("baseline_lam")   # lambda-independent outputs (baseline + sweep)
def out(pair, name): return f"{S.out_root}/{pair[0]}_{pair[1]}/{'' if SHARED(name) else OP + '/'}{name}.json"
def done(pair, name): return os.path.exists(out(pair, name))
def load(pair, name): return json.load(open(out(pair, name)))
def ckpt(pair, name): return f"{S.out_root}/ckpt/{pair[0]}_{pair[1]}_{name if name == 'baseline' else OP + '_' + name}.pt"
save_json(S.__dict__, f"{S.out_root}/settings_{OP}.json")
def lname(lam): return f"lam{lam:.0e}"
def ntr(lam): return S.n_trials if any(abs(lam - q) / q < 1e-6 for q in PRIMARY) else N2

def data_for(ds):
    train_ds, train_eval, val, test = get_datasets(ds, S.data_root, S.train_seed)
    nc = NUM_CLASSES[ds]
    ev = stratified_subset(test, S.eval_subset, nc, 0)
    if S.train_subset:   # smoke budget only
        train_ds = torch.utils.data.Subset(train_ds, list(range(S.train_subset))); train_eval = torch.utils.data.Subset(train_eval, list(range(S.train_subset)))
        val = torch.utils.data.Subset(val, list(range(min(len(val), S.train_subset)))); test = stratified_subset(test, S.train_subset, nc, 1)
    return dict(train=loader(train_ds, S.batch, True, S.workers), train_eval=loader(train_eval, S.batch, False, S.workers),
                val=loader(val, S.batch, False, S.workers), test=loader(test, S.batch, False, S.workers),
                eval=preload(ev, S.batch), nc=nc, n_eval=len(ev))

def fresh(pair): seed_everything(S.train_seed); return build_model(pair[0], NUM_CLASSES[pair[1]]).to(DEVICE)
def load_model(pair, name):
    m = fresh(pair); m.load_state_dict(torch.load(ckpt(pair, name), map_location=DEVICE)); return m.eval()

def mc_record(pair, name, model, D, lam, n, dist="uniform", clip=None, dup=None, protected=None, extra=None):
    """Run Algorithm 1 for one configuration and store per-seed outputs. clip = bounds dict, dup = critical blocks list."""
    if done(pair, name):
        prev = load(pair, name)
        if prev.get("n_trials", 0) >= n: print(f"   [skip] {name}"); return prev
        print(f"   [redo] {name}: {prev.get('n_trials')} trials stored, {n} required")
    model.eval()
    if dup: apply_selective_redundancy(model, dup)
    ctx = RangeRestriction(model, clip, S.logit_clip) if clip else _Null()
    with ctx:
        clean = accuracy(model, D["eval"]); clean_full = accuracy(model, D["test"]) if D["n_eval"] != len(D["test"].dataset) else clean
        if protected is not None: r = monte_carlo_protected(model, D["eval"], lam, n, clean, protected, dist, desc=name)
        else: r = monte_carlo(model, D["eval"], lam, n, clean, dist, desc=name)
    if dup: remove_selective_redundancy(model)
    r.update({"config": name, "arch": pair[0], "dataset": pair[1], "clean_acc_full_test": clean_full, "eval_images": D["n_eval"], "bit_dist": dist, **(extra or {})})
    save_json(r, out(pair, name)); print(f"   {name}: clean {clean*100:.2f}  CER {r['cer_mean']*100:.3f} ± {r['cer_se']*100:.3f} pp", flush=True)
    return r

class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): pass

# ============================================================================= stages
for pair in PAIRS + [(a, "cifar10") for a in args.extra_archs.split(",") if a]:
    arch, ds = pair; tag = f"{arch}/{ds}"; print(f"\n================ {tag} ================", flush=True)
    D = data_for(ds); full = pair in FULL; extra = arch in args.extra_archs.split(",")

    # ---------------- baseline training
    if "baseline" in STAGES and not done(pair, "train_baseline"):
        m = fresh(pair); h = train(m, D["train"], D["val"], S, ckpt=ckpt(pair, "baseline"), tag=f"{tag} baseline")
        h.update({"test_acc": accuracy(m, D["test"]), "params": sum(p.numel() for p in m.parameters())}); save_json(h, out(pair, "train_baseline"))
    if not os.path.exists(ckpt(pair, "baseline")): print("baseline checkpoint missing; run stage 'baseline' first"); continue
    base = load_model(pair, "baseline")

    # ---------------- baseline vulnerability (Table 3, Fig. 2) and Fig. 5 lambda sweep
    if "vuln" in STAGES:
        for lam in SWEEP: mc_record(pair, f"baseline_{lname(lam)}", base, D, lam, ntr(lam))
    if extra:
        if "fat" in STAGES and not done(pair, "train_fat"):
            m = fresh(pair); h = train(m, D["train"], D["val"], S, fat_lam_max=S.lam_fat_max, ckpt=ckpt(pair, "fat"), tag=f"{tag} FAT")
            h["test_acc"] = accuracy(m, D["test"]); save_json(h, out(pair, "train_fat"))
        if "fat" in STAGES: mc_record(pair, f"fat_{lname(S.lam_op)}", load_model(pair, "fat"), D, S.lam_op, S.n_trials)
        continue

    if not full and ds != "cifar10": continue            # EfficientNet-B0 / GTSRB: baseline vulnerability only (Table 3, Fig. 2)
    if not full:                                          # EfficientNet-B0 / CIFAR-10: curriculum FAT only (Table 5)
        if "fat" in STAGES:
            if not done(pair, "train_fat"):
                m = fresh(pair); h = train(m, D["train"], D["val"], S, fat_lam_max=S.lam_fat_max, ckpt=ckpt(pair, "fat"), tag=f"{tag} FAT")
                h["test_acc"] = accuracy(m, D["test"]); save_json(h, out(pair, "train_fat"))
            fat = load_model(pair, "fat")
            for lam in SWEEP: mc_record(pair, f"fat_{lname(lam)}", fat, D, lam, ntr(lam))
        continue

    # ---------------- layer-wise sensitivity (Fig. 3) and critical set
    if "sens" in STAGES and not done(pair, "sensitivity"):
        clean = accuracy(base, D["eval"])
        Sk = layer_sensitivity(base, D["eval"], S.lam_sens, S.n_sens_trials, clean)
        C, T_used = critical_set(Sk, S.T_pp, S.critical_rule)
        save_json({"S_k_pp": Sk, "T_pp": T_used, "rule": S.critical_rule, "critical": C, "n_trials_per_layer": S.n_sens_trials, "lambda": S.lam_sens}, out(pair, "sensitivity"))
    if not done(pair, "sensitivity"): continue
    sens = load(pair, "sensitivity"); Sk, C = sens["S_k_pp"], sens["critical"]
    print(f"   critical blocks (S_k >= {sens['T_pp']:.3f} pp, rule {sens.get('rule')}): {C}", flush=True)

    # ---------------- architecture-only (Table 4, Fig. 4): clip, duplicate-and-compare, residual regularisation
    if "arch" in STAGES:
        bounds = calibrate_bounds(base, D["train_eval"], S.calib_images, S.clip_percentile, S.clip_factor)
        save_json(bounds, out(pair, "activation_bounds"))
        macs = layer_macs(base); dup_layers = [n for n in injectable_layers(base) if block_of(n) in C]
        ovh = cost_change(macs, {}, {}, dup_layers)
        mc_record(pair, "arch_clip_only", base, D, S.lam_op, N2, clip=bounds)
        mc_record(pair, "arch_dup_only", base, D, S.lam_op, N2, dup=C, extra={"overhead": ovh})
        if hasattr(base, "residual_blocks"):
            if not done(pair, "train_archreg"):
                m = fresh(pair); h = train(m, D["train"], D["val"], S, beta_res=S.beta_res, ckpt=ckpt(pair, "archreg"), tag=f"{tag} beta-reg")
                h["test_acc"] = accuracy(m, D["test"]); save_json(h, out(pair, "train_archreg"))
            reg = load_model(pair, "archreg"); mc_record(pair, "arch_reg_only", reg, D, S.lam_op, N2)
        else: reg = base
        bounds_reg = calibrate_bounds(reg, D["train_eval"], S.calib_images, S.clip_percentile, S.clip_factor)
        mc_record(pair, "arch_all", reg, D, S.lam_op, S.n_trials, clip=bounds_reg, dup=C, extra={"overhead": ovh, "critical": C})

    # ---------------- training-only: curriculum FAT (Table 5, Fig. 5)
    if "fat" in STAGES:
        if not done(pair, "train_fat"):
            m = fresh(pair); h = train(m, D["train"], D["val"], S, fat_lam_max=S.lam_fat_max, ckpt=ckpt(pair, "fat"), tag=f"{tag} FAT")
            h["test_acc"] = accuracy(m, D["test"]); save_json(h, out(pair, "train_fat"))
        fat = load_model(pair, "fat")
        for lam in SWEEP: mc_record(pair, f"fat_{lname(lam)}", fat, D, lam, ntr(lam))

    if not full: continue

    # ---------------- optimisation-only (Fig. 6, Table 6)
    if "opt" in STAGES:
        act_scales = calibrate_act_scales(base, D["train_eval"], S.mp_calib_images)
        for pol in ["MP-1", "MP-2", "MP-3", "MP-4", "MP-5"]:            # Table 6 on the unpruned baseline
            prec = precision_policy(base, pol, C); mq = apply_mixed_precision(base, prec, act_scales)
            mc_record(pair, f"mp_{pol}", mq, D, S.lam_op, S.n_trials, extra={"policy": MP_POLICIES[pol], "cost": cost_of(base, prec)})
        for scale in (1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3):                  # sparsity sweep, Fig. 6 (scale 1.0 = 30 % global, Table 7)
            name = f"prune_s{scale:.2f}"
            if not os.path.exists(ckpt(pair, name)):
                m = load_model(pair, "baseline"); m, masks, log = sensitivity_aware_pruning(m, Sk, S, D["train"], D["val"], scale, tag=f"{tag} {name}")
                torch.save({"state": m.state_dict(), "masks": {k: v.cpu() for k, v in masks.items()}, "log": log}, ckpt(pair, name))
            ck = torch.load(ckpt(pair, name), map_location=DEVICE); m = fresh(pair); m.load_state_dict(ck["state"]); m.eval()
            masks = {k: v.to(DEVICE) for k, v in ck["masks"].items()}; sp = ck["log"]["final_sparsity"]
            n = S.n_trials if abs(scale - 1.0) < 1e-6 else N2
            mc_record(pair, name, m, D, S.lam_op, n, extra={"sparsity": sp, "cost": cost_of(m, {}, masks), "prune_log": ck["log"]})
            prec = precision_policy(m, "MP-3", C); mq = apply_mixed_precision(m, prec, calibrate_act_scales(m, D["train_eval"], S.mp_calib_images))
            mc_record(pair, name + "_MP3", mq, D, S.lam_op, n, extra={"sparsity": sp, "cost": cost_of(m, prec, masks)})

    # ---------------- unified framework (Table 7 / Table 8, Fig. 7 / Fig. 8)
    if "unified" in STAGES:
        if not os.path.exists(ckpt(pair, "unified")):
            if not done(pair, "train_unified"):          # hardened (beta) + curriculum FAT from scratch
                m = fresh(pair); h = train(m, D["train"], D["val"], S, fat_lam_max=S.lam_fat_max, beta_res=S.beta_res, ckpt=ckpt(pair, "unified_pre"), tag=f"{tag} unified train")
                h["test_acc"] = accuracy(m, D["test"]); save_json(h, out(pair, "train_unified"))
            m = load_model(pair, "unified_pre"); m, masks, log = sensitivity_aware_pruning(m, Sk, S, D["train"], D["val"], 1.0, fat_lam_max=S.lam_fat_max, tag=f"{tag} unified prune")
            torch.save({"state": m.state_dict(), "masks": {k: v.cpu() for k, v in masks.items()}, "log": log}, ckpt(pair, "unified"))
        ck = torch.load(ckpt(pair, "unified"), map_location=DEVICE); m = fresh(pair); m.load_state_dict(ck["state"]); m.eval()
        masks = {k: v.to(DEVICE) for k, v in ck["masks"].items()}
        bounds_u = calibrate_bounds(m, D["train_eval"], S.calib_images, S.clip_percentile, S.clip_factor)
        prec = precision_policy(m, "MP-3", C); mq = apply_mixed_precision(m, prec, calibrate_act_scales(m, D["train_eval"], S.mp_calib_images))
        cost = cost_of(m, prec, masks, [n for n in injectable_layers(m) if block_of(n) in C])
        mc_record(pair, "unified", mq, D, S.lam_op, S.n_trials, clip=bounds_u, dup=C, extra={"sparsity": ck["log"]["final_sparsity"], "cost": cost, "critical": C})
        # paired tests, family of four ablations vs baseline (Table 7)
        b = load(pair, f"baseline_{lname(S.lam_op)}")["cer_per_seed"]; tests = {}
        for cfg in ["arch_all", f"fat_{lname(S.lam_op)}", "prune_s1.00_MP3", "unified"]:
            if done(pair, cfg): tests[cfg] = paired_z(b, load(pair, cfg)["cer_per_seed"], m=4)
        save_json(tests, out(pair, "ablation_tests"))

    # ---------------- comparators (Table 10) -- CIFAR-10 only
    if "comparators" in STAGES and ds == "cifar10":
        bounds = json.load(open(out(pair, "activation_bounds"))) if done(pair, "activation_bounds") else calibrate_bounds(base, D["train_eval"], S.calib_images, S.clip_percentile, S.clip_factor)
        mc_record(pair, "cmp_range_restriction", base, D, S.lam_op, S.n_trials, clip=bounds, extra={"cost": {"latency_pct": 1.1, "memory_pct": 0.0, "energy_pct": 1.1}})
        if not done(pair, "train_fat_rabbit"):
            m = fresh(pair); h = train_fat_rabbit(m, D["train"], D["val"], S, S.lam_op, ckpt(pair, "fat_rabbit")); h["test_acc"] = accuracy(m, D["test"]); save_json(h, out(pair, "train_fat_rabbit"))
        mc_record(pair, "cmp_fat_rabbit", load_model(pair, "fat_rabbit"), D, S.lam_op, S.n_trials)
        if not os.path.exists(ckpt(pair, "rap")):
            m = load_model(pair, "baseline"); m, masks, log = rap_pruning(m, Sk, S, D["train"], D["val"], S.lam_op, 1.0)
            torch.save({"state": m.state_dict(), "masks": {k: v.cpu() for k, v in masks.items()}, "log": log}, ckpt(pair, "rap"))
        ck = torch.load(ckpt(pair, "rap"), map_location=DEVICE); m = fresh(pair); m.load_state_dict(ck["state"]); m.eval()
        mc_record(pair, "cmp_rap", m, D, S.lam_op, S.n_trials, extra={"sparsity": ck["log"]["final_sparsity"], "cost": cost_of(m, {}, {k: v.to(DEVICE) for k, v in ck["masks"].items()})})
        if not done(pair, "rl_policy"):
            clean = accuracy(base, D["eval"]); blocks = list(Sk.keys())
            chosen, log = rl_bit_protection(base, blocks, D["eval"], S, S.lam_op, clean, budget=4, episodes=40 if S.budget != "smoke" else 3, eval_trials=8 if S.budget != "smoke" else 1)
            save_json({"chosen": chosen, **log, "overhead": protection_overhead(base, chosen)}, out(pair, "rl_policy"))
        rl = load(pair, "rl_policy")
        mc_record(pair, "cmp_rl_bit_protection", base, D, S.lam_op, S.n_trials, protected=[tuple(x) for x in rl["chosen"]], extra={"cost": rl["overhead"], "chosen": rl["chosen"]})
        if done(pair, "unified"):
            u = load(pair, "unified")["cer_per_seed"]; tests = {}
            for cfg in [f"baseline_{lname(S.lam_op)}", "cmp_range_restriction", "cmp_fat_rabbit", "cmp_rap", "cmp_rl_bit_protection"]:
                if done(pair, cfg): tests[cfg] = paired_z(load(pair, cfg)["cer_per_seed"], u, m=4)
            save_json(tests, out(pair, "comparator_tests"))

    # ---------------- bit-position distribution sensitivity (Table 11) -- CIFAR-10 only
    if "bitdist" in STAGES and ds == "cifar10" and os.path.exists(ckpt(pair, "unified")):
        ck = torch.load(ckpt(pair, "unified"), map_location=DEVICE); m = fresh(pair); m.load_state_dict(ck["state"]); m.eval()
        bounds_u = calibrate_bounds(m, D["train_eval"], S.calib_images, S.clip_percentile, S.clip_factor)
        mq = apply_mixed_precision(m, precision_policy(m, "MP-3", C), calibrate_act_scales(m, D["train_eval"], S.mp_calib_images))
        for dist in ("exponent", "mantissa"):
            mc_record(pair, f"baseline_{dist}", base, D, S.lam_op, S.n_trials, dist=dist)
            mc_record(pair, f"unified_{dist}", mq, D, S.lam_op, S.n_trials, dist=dist, clip=bounds_u, dup=C)

print("\npipeline finished", flush=True)
