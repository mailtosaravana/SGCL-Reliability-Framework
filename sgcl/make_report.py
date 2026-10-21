#!/usr/bin/env python3
"""
Build every table (3-8, 10, 11) and figure (2-8) of the manuscript from the JSON outputs of run_pipeline.py.

    python make_report.py --results results --out report

Writes report/tables/*.csv + *.md, report/figures/fig*.png|pdf, report/summary.json (every number quoted in the text)
and report/per_seed/*.csv (per-seed accuracy and CER of every configuration, seeds 0..N-1).
"""
from __future__ import annotations
import os, sys, json, glob, argparse, math, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgcl.stats import paired_z, fmt_p

p = argparse.ArgumentParser(); p.add_argument("--results", default="results"); p.add_argument("--out", default="report")
p.add_argument("--lam-op", type=float, default=None, help="operating point to report (default: the one in results/settings.json)")
args = p.parse_args(); R, O = args.results, args.out
os.makedirs(f"{O}/tables", exist_ok=True); os.makedirs(f"{O}/figures", exist_ok=True); os.makedirs(f"{O}/per_seed", exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

ARCH = {"resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0", "vgg16": "VGG-16", "mobilenet_v2": "MobileNetV2"}
DS = {"cifar10": "CIFAR-10", "gtsrb": "GTSRB"}
settings = json.load(open(f"{R}/settings.json")) if os.path.exists(f"{R}/settings.json") else {}
LAM = args.lam_op or settings.get("lam_op", 1e-5); LAM_OP = f"lam{LAM:.0e}"
if os.path.exists(f"{R}/settings_op{LAM:.0e}.json"): settings = json.load(open(f"{R}/settings_op{LAM:.0e}.json"))
summary = {"budget": settings.get("budget"), "settings": settings}

OP = f"op{LAM:.0e}"
def J(pair, name):
    for f in (f"{R}/{pair}/{OP}/{name}.json", f"{R}/{pair}/{name}.json"):
        if os.path.exists(f): return json.load(open(f))
    return None
def pairs(): return sorted(os.path.basename(d) for d in glob.glob(f"{R}/*_*") if os.path.isdir(d) and os.path.basename(d) != "ckpt")
def split_pair(pr):
    for d in DS:
        if pr.endswith("_" + d): return pr[:-len(d) - 1], d
    return pr.rsplit("_", 1)
def pct(x, d=2): return "–" if x is None else f"{100*x:.{d}f}"
def lamtxt(l): e = int(math.floor(math.log10(l) + 1e-9)); m = l / 10**e; return f"10^{e}" if abs(m - 1) < 1e-6 else f"{m:.0f}×10^{e}"
SWEEP = sorted(float(os.path.basename(f)[len("baseline_lam"):-5]) for f in glob.glob(f"{R}/resnet18_cifar10/baseline_lam*.json")) or [1e-6, 1e-5, 1e-4]
OPTXT = lamtxt(LAM)
def write_table(name, header, rows, caption):
    with open(f"{O}/tables/{name}.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(f"{O}/tables/{name}.md", "w") as f:
        f.write(f"**{caption}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows: f.write("| " + " | ".join(str(c) for c in r) + " |\n")
    print(f"  wrote {name}: {len(rows)} rows")
def savefig(name):
    plt.tight_layout(); plt.savefig(f"{O}/figures/{name}.png", dpi=300); plt.savefig(f"{O}/figures/{name}.pdf"); plt.close(); print(f"  wrote {name}")
def per_seed_csv(pair, name, r):
    with open(f"{O}/per_seed/{pair}_{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "accuracy", "cer"])
        for i, (a, c) in enumerate(zip(r["acc_per_seed"], r["cer_per_seed"])): w.writerow([i, f"{a:.6f}", f"{c:.6f}"])

# dump per-seed outputs of every configuration
for pr in pairs():
    for f in glob.glob(f"{R}/{pr}/*.json") + glob.glob(f"{R}/{pr}/{OP}/*.json"):
        r = json.load(open(f))
        if isinstance(r, dict) and "acc_per_seed" in r: per_seed_csv(pr, os.path.basename(f)[:-5], r)

# ----------------------------------------------------------------------------- Table 3 + Figure 2: baseline vulnerability
rows = []; fig, ax = plt.subplots(figsize=(4.2, 3))
for pr in pairs():
    a, d = split_pair(pr)
    if a not in ("resnet18", "efficientnet_b0"): continue
    tb = J(pr, "train_baseline"); b0 = J(pr, f"baseline_{LAM_OP}")
    if not b0: continue
    clean = b0["clean_acc_full_test"]; rows.append([f"{ARCH[a]} / {DS[d]}", "0", pct(clean), pct(1 - clean), "0.00"])
    xs, ys, es = [], [], []
    for lam in SWEEP:
        r = J(pr, f"baseline_lam{lam:.0e}")
        if not r: continue
        acc = clean - r["cer_mean"]           # faulted accuracy on the full test split = clean(full) - CER(eval subset)
        if any(abs(lam - q) / q < 1e-6 for q in (LAM / 10, LAM, LAM * 10)): rows.append(["", lamtxt(lam), pct(acc), pct(1 - acc), pct(r["cer_mean"])])
        xs.append(lam); ys.append(100 * acc); es.append(100 * r["cer_se"])
        summary[f"{pr}/baseline_cer_lam{lam:.0e}"] = r["cer_mean"]
    summary[f"{pr}/clean_acc"] = clean
    if xs: ax.errorbar(xs, ys, es, marker="o", ms=3, capsize=2, label=f"{ARCH[a]} / {DS[d]}")
ax.set_xscale("log"); ax.set_xlabel("Bit-flip injection probability λ"); ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=7); savefig("fig2_baseline_vs_lambda")
write_table("table3_baseline_vulnerability", ["Model / dataset", "λ", "Accuracy (%)", "Misclassification rate (%)", "CER (%)"], rows,
            "Table 3. Misclassification rate and critical error rate of the baseline models on CIFAR-10 and GTSRB.")

# ----------------------------------------------------------------------------- Figure 3: layer-wise sensitivity (ResNet-18 / CIFAR-10)
s = J("resnet18_cifar10", "sensitivity")
if s:
    ks = list(s["S_k_pp"].keys()); vs = [s["S_k_pp"][k] for k in ks]
    fig, ax = plt.subplots(figsize=(4.8, 2.8)); cols = ["#c0392b" if v >= s["T_pp"] else "#7f8c8d" for v in vs]
    ax.bar(ks, vs, color=cols); ax.axhline(s["T_pp"], ls="--", c="k", lw=0.8); ax.text(0.1, s["T_pp"] + 0.1, f"T = {s['T_pp']} pp", fontsize=7)
    ax.set_ylabel("Sensitivity $S_k$ (pp)"); plt.xticks(rotation=45, ha="right"); savefig("fig3_layer_sensitivity")
    summary["resnet18_cifar10/S_k_pp"] = s["S_k_pp"]; summary["resnet18_cifar10/critical"] = s["critical"]

# ----------------------------------------------------------------------------- Table 4 + Figure 4: architectural hardening
rows = []; labels, bcer, hcer = [], [], []
for pr in pairs():
    a, d = split_pair(pr)
    if d != "cifar10" or a not in ("resnet18", "efficientnet_b0"): continue
    b = J(pr, f"baseline_{LAM_OP}"); h = J(pr, "arch_all")
    if not (b and h): continue
    ovh = h.get("overhead", {}); rif = b["cer_mean"] / h["cer_mean"] if h["cer_mean"] > 0 else float("inf")
    rows.append([ARCH[a], pct(b["cer_mean"]), pct(h["cer_mean"]), f"{rif:.2f}×", f"{ovh.get('latency_pct', 0):+.1f}"])
    labels.append(ARCH[a]); bcer.append(100 * b["cer_mean"]); hcer.append(100 * h["cer_mean"]); summary[f"{pr}/arch_all_rif"] = rif
    for cfg in ("arch_clip_only", "arch_dup_only", "arch_reg_only"):
        r = J(pr, cfg)
        if r: summary[f"{pr}/{cfg}_cer"] = r["cer_mean"]
if rows:
    write_table("table4_architectural_hardening", ["Model", "Baseline CER (%)", "Hardened CER (%)", "RIF", "Inference overhead (%)"], rows,
                f"Table 4. RIF and inference overhead of architectural hardening (CIFAR-10, λ = {OPTXT}).")
    fig, ax = plt.subplots(figsize=(3.6, 2.8)); x = np.arange(len(labels)); ax.bar(x - 0.18, bcer, 0.36, label="Baseline"); ax.bar(x + 0.18, hcer, 0.36, label="Hardened")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("CER (%)"); ax.legend(fontsize=7); savefig("fig4_architectural_hardening")

# ----------------------------------------------------------------------------- Table 5 + Figure 5: fault-aware training
rows = []; fig, ax = plt.subplots(figsize=(4.2, 3))
for pr in pairs():
    a, d = split_pair(pr)
    if d != "cifar10": continue
    b = J(pr, f"baseline_{LAM_OP}"); f_ = J(pr, f"fat_{LAM_OP}"); tb = J(pr, "train_baseline"); tf = J(pr, "train_fat")
    if not (b and f_): continue
    vr = 100 * (1 - f_["acc_var"] / b["acc_var"]) if b["acc_var"] > 0 else float("nan")
    sup = 100 * (1 - f_["cer_mean"] / b["cer_mean"]) if b["cer_mean"] > 0 else float("nan")
    tovh = 100 * (tf["train_time_s"] / tb["train_time_s"] - 1) if (tb and tf and tb.get("train_time_s")) else float("nan")
    rows.append([ARCH[a], f"{b['acc_var']:.2e}", f"{f_['acc_var']:.2e}", f"{vr:.1f}", pct(b["cer_mean"]), pct(f_["cer_mean"]), f"{sup:.1f}", f"{tovh:+.0f}"])
    summary[f"{pr}/fat_rif"] = b["cer_mean"] / f_["cer_mean"] if f_["cer_mean"] > 0 else None
    if a == "resnet18":
        for cfg, lab in (("baseline", "Baseline"), ("fat", "Curriculum FAT")):
            xs, ys, es = [], [], []
            for lam in SWEEP:
                r = J(pr, f"{cfg}_lam{lam:.0e}")
                if r: xs.append(lam); ys.append(100 * (r["clean_acc_full_test"] - r["cer_mean"])); es.append(100 * math.sqrt(r["acc_var"]))
            if xs: ax.errorbar(xs, ys, es, marker="o", ms=3, capsize=2, label=lab)
ax.set_xscale("log"); ax.set_xlabel("Bit-flip injection probability λ"); ax.set_ylabel("Accuracy (%)  (±1 SD over seeds)"); ax.legend(fontsize=7); savefig("fig5_fat_vs_lambda")
if rows: write_table("table5_fault_aware_training", ["Model", "Acc. variance (baseline)", "Acc. variance (FAT)", "Variance reduction (%)", "CER baseline (%)", "CER FAT (%)", "CER suppression (%)", "Training overhead (%)"], rows,
                     "Table 5. Variance reduction and CER suppression by fault-aware training on CIFAR-10.")

# ----------------------------------------------------------------------------- Table 6 + Figure 6: optimisation (ResNet-18 / CIFAR-10)
pr = "resnet18_cifar10"; b = J(pr, f"baseline_{LAM_OP}")
if b:
    rows = [["Baseline", "Uniform FP32 (reference)", pct(b["cer_mean"]), "0.0", "0.0", "0.0", "0.0"]]
    for pol in ["MP-1", "MP-2", "MP-3", "MP-4", "MP-5"]:
        r = J(pr, f"mp_{pol}")
        if not r: continue
        c = r["cost"]; rows.append([pol, r["policy"], pct(r["cer_mean"]), f"{100*(r['cer_mean']/b['cer_mean']-1):+.1f}", f"{c['latency_pct']:+.1f}", f"{c['memory_pct']:+.1f}", f"{c['energy_pct']:+.1f}"])
        summary[f"{pr}/mp_{pol}_cer"] = r["cer_mean"]
    write_table("table6_mixed_precision", ["ID", "Precision policy", "CER (%)", "ΔCER vs baseline (rel. %)", "Latency change (%)", "Memory change (%)", "Energy change (%)"], rows,
                f"Table 6. Mixed-precision allocation policies (ResNet-18, CIFAR-10, λ = {OPTXT}).")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for suf, lab in (("", "Pruning only"), ("_MP3", "Pruning + MP-3")):
        xs, ys = [], []
        for sc in (1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3):
            r = J(pr, f"prune_s{sc:.2f}{suf}")
            if r: xs.append(100 * r["sparsity"]); ys.append(b["cer_mean"] / r["cer_mean"] if r["cer_mean"] > 0 else float("nan"))
        if xs: ax.plot(xs, ys, marker="o", ms=3, label=lab)
    ax.set_xlabel("Global sparsity (%)"); ax.set_ylabel("RIF"); ax.legend(fontsize=7); savefig("fig6_rif_vs_sparsity")

# ----------------------------------------------------------------------------- Table 7 + Figure 7 (CIFAR-10), Table 8 + Figure 8 (GTSRB)
CFG = [(f"baseline_{LAM_OP}", "Baseline"), ("arch_all", "Architecture-only"), (f"fat_{LAM_OP}", "Training-only"), ("prune_s1.00_MP3", "Optimization-only"), ("unified", "Unified framework")]
for pr, tab, fig_name in (("resnet18_cifar10", "table7_ablation", "fig7_ablation_rif"), ("resnet18_gtsrb", "table8_gtsrb", "fig8_gtsrb_misclassification")):
    b = J(pr, f"baseline_{LAM_OP}"); t = J(pr, "ablation_tests")
    if not b: continue
    rows = []; labs, rifs, mis = [], [], []
    for cfg, lab in CFG:
        r = J(pr, cfg)
        if not r: continue
        if cfg.startswith("baseline"): st = None
        else: st = paired_z(b["cer_per_seed"], r["cer_per_seed"], 4)      # always recomputed from the stored per-seed CER
        rif = b["cer_mean"] / r["cer_mean"] if r["cer_mean"] > 0 else float("inf")
        labs.append(lab); rifs.append(rif); mis.append(100 * (1 - (r["clean_acc_full_test"] - r["cer_mean"])))
        summary[f"{pr}/{cfg}"] = {"cer": r["cer_mean"], "clean_full": r["clean_acc_full_test"], "rif": rif, "test": st}
        if pr.endswith("cifar10"):
            rows.append([lab, pct(r["cer_mean"]), "–" if st is None else f"{st['delta_cer_pp']:.2f}", "–" if st is None else f"{st['z']:.2f}",
                         "–" if st is None else fmt_p(st["p_adj"]), "–" if st is None else f"[{st['ci95_pp'][0]:.2f}, {st['ci95_pp'][1]:.2f}]", f"{rif:.2f}×"])
        else:
            fph = r["cer_mean"] * 10 * 3600
            rows.append([lab, OPTXT, "10 Hz", pct(r["cer_mean"]), f"{fph:.1f}", f"{r['cer_mean']*1e9:.2e}", f"{rif:.3f}×",
                         "–" if st is None else f"z = {st['z']:.2f}, adj. p = {fmt_p(st['p_adj'])}"])
    if pr.endswith("cifar10"):
        write_table(tab, ["Configuration", "CER (%)", "Absolute CER reduction (pp)", "z-score", "p (Bonferroni, m = 4)", "95 % CI (pp)", "RIF"], rows,
                    f"Table 7. Ablation of the proposed framework with paired statistical tests (ResNet-18, CIFAR-10, λ = {OPTXT}).")
        fig, ax = plt.subplots(figsize=(3.8, 2.8)); ax.bar(labs, rifs, color="#2c3e50"); ax.set_ylabel("RIF"); plt.xticks(rotation=30, ha="right")
        for i, v in enumerate(rifs): ax.text(i, v + 0.03, f"{v:.2f}×", ha="center", fontsize=7)
        savefig(fig_name)
    else:
        write_table(tab, ["Configuration", "λ", "Decision rate", "CER (%)", "Failures / hour", "Fault-induced misclassifications per 10⁹ decisions", "Reduction factor (RIF)", "Paired test vs baseline"], rows,
                    f"Table 8. Application-level functional-failure rates on GTSRB (ResNet-18, λ = {OPTXT}).")
        fig, ax = plt.subplots(figsize=(3.8, 2.8)); ax.bar(labs, mis, color="#16a085"); ax.axhline(100 * (1 - b["clean_acc_full_test"]), ls="--", c="k", lw=0.8, label="clean")
        ax.set_ylabel("Misclassification rate (%)"); plt.xticks(rotation=30, ha="right"); ax.legend(fontsize=7); savefig(fig_name)

# ----------------------------------------------------------------------------- Table 10: matched comparators (ResNet-18 / CIFAR-10)
pr = "resnet18_cifar10"; u = J(pr, "unified"); t = J(pr, "comparator_tests")
if u:
    b = J(pr, f"baseline_{LAM_OP}"); rows = []
    for cfg, lab, layer in ((f"baseline_{LAM_OP}", "Baseline (unprotected)", "–"), ("cmp_range_restriction", "Range restriction [34]", "Architecture"),
                            ("cmp_fat_rabbit", "FAT-RABBIT [35]", "Training"), ("cmp_rap", "RAP [40]", "Optimization (pruning)"),
                            ("cmp_rl_bit_protection", "RL bit protection [36]", "Bit-level protection"), ("unified", "Proposed unified framework", "Cross-layer")):
        r = J(pr, cfg)
        if not r: continue
        rif = b["cer_mean"] / r["cer_mean"] if r["cer_mean"] > 0 else float("inf"); cost = r.get("cost", {}).get("latency_pct", 0.0)
        st = None if cfg == "unified" else paired_z(r["cer_per_seed"], u["cer_per_seed"], 4)
        rows.append([lab, layer, pct(r["clean_acc_full_test"]), pct(r["cer_mean"]), f"{rif:.2f}×", f"{cost:+.1f}",
                     "–" if st is None else f"{st['delta_cer_pp']:.2f} [{st['ci95_pp'][0]:.2f}, {st['ci95_pp'][1]:.2f}]", "–" if st is None else f"{st['z']:.2f} / {fmt_p(st['p_adj'])}"])
        summary[f"{pr}/{cfg}_cmp"] = {"cer": r["cer_mean"], "clean_full": r["clean_acc_full_test"], "rif": rif, "cost_latency_pct": cost, "test": st}
    write_table("table10_matched_comparison", ["Method", "Layer of the reliability stack", "Clean accuracy (%)", f"CER at λ = {OPTXT} (%)", "RIF", "Inference-cost change (%)", "ΔCER vs unified (pp) [95 % CI]", "z / adjusted p"], rows,
                f"Table 10. Matched comparison with re-implemented methods (ResNet-18, CIFAR-10, λ = {OPTXT}).")

# ----------------------------------------------------------------------------- Table 11: bit-position distribution
rows = []
for dist, lab in (("uniform", "Uniform (1/32, 8/32, 23/32)"), ("exponent", "Exponent-weighted (0.05 / 0.60 / 0.35)"), ("mantissa", "Mantissa-weighted (0.05 / 0.10 / 0.85)")):
    b = J(pr, f"baseline_{LAM_OP}" if dist == "uniform" else f"baseline_{dist}"); un = J(pr, "unified" if dist == "uniform" else f"unified_{dist}")
    if b and un:
        rif = b["cer_mean"] / un["cer_mean"] if un["cer_mean"] > 0 else float("inf")
        rows.append([lab, pct(b["cer_mean"]), pct(un["cer_mean"]), f"{rif:.2f}×"]); summary[f"{pr}/bitdist_{dist}"] = {"baseline_cer": b["cer_mean"], "unified_cer": un["cer_mean"], "rif": rif}
if rows: write_table("table11_bit_distribution", ["Bit-position distribution (sign / exponent / mantissa)", "Baseline CER (%)", "Unified CER (%)", "RIF"], rows,
                     f"Table 11. Sensitivity of CER and RIF to the bit-position distribution (ResNet-18, CIFAR-10, λ = {OPTXT}).")

json.dump(summary, open(f"{O}/summary.json", "w"), indent=1, default=str)
print(f"report written to {O}/")
