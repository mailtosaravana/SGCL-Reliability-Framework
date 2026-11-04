# Reproduction Guide for Reviewers

This archive contains all artifacts, checkpoints, configurations, and raw per-seed trial logs (500 paired seeds) for the manuscript experiments across ResNet-18 and EfficientNet-B0 on CIFAR-10 and GTSRB.

## 1. Quick Reproduction of Reported Tables and Figures
To re-generate all manuscript tables and figures directly from the raw 500-trial evaluation outputs:

```bash
cd sgcl
python make_report.py --results ../results --out ../report_lam1e-07 --lam-op 1e-07
```

## 2. Parameter and Sparsity Verification
To verify the exact layer-wise sparsity, residual-block sparsity, and whole-model sparsity:

```bash
python sgcl/tools/report_sparsity.py results/ckpt/resnet18_cifar10_op1e-07_unified.pt
```

## 3. Artifact Integrity Verification
Verify file integrity using the SHA-256 manifest:

```bash
python sgcl/tools/repro_manifest.py --results ../results --report ../report_lam1e-07 --check MANIFEST.json
```