# Reproducibility manifest

- budget: **fast**, epochs 20, FT epochs 3, eval subset 1000, N = 500
- lam_op 1e-07, lam_fat_max 2e-07, lam_sens 1e-05, critical rule median
- code commit: `None` (dirty: False)
- report command: `python make_report.py --results /content/drive/MyDrive/sgcl_results_v3/results --out report_lam1e-07 --lam-op 1e-07`

| Table / figure | pairs | raw files |
|---|---|---|
| Table 3 / Fig. 2 | all four pairs | 40 |
| Table 4 / Fig. 4 | resnet18_cifar10 (+ extra archs) | 44 |
| Table 5 / Fig. 5 | CIFAR-10 pairs | 70 |
| Table 6 / Fig. 6 | resnet18_cifar10 | 30 |
| Table 7 / Figs. 7-8 | resnet18_cifar10 | 8 |
| Table 8 / Fig. 8 | resnet18_gtsrb | 40 |
| Table 10 | resnet18_cifar10 | 7 |
| Table 11 | resnet18_cifar10 | 4 |
| Fig. 3 | resnet18 pairs | 2 |

Per-file SHA-256 and trial counts: see MANIFEST.json.
