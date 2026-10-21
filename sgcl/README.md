# SGCL — Sensitivity-Guided Cross-Layer Soft-Error Resilience Framework

Complete, runnable PyTorch implementation of every experiment in

> *A Unified Sensitivity Guided Cross Layer Framework for Soft Error Resilience in Machine Learning Systems*

The pipeline trains the models, runs the structured Monte Carlo bit-flip fault-injection protocol
(Section III-C/III-D, Algorithm 1), estimates layer-wise sensitivity (III-F), applies the architectural
hardening (III-G), curriculum fault-aware training (III-H), sensitivity-aware pruning and mixed precision
(III-I), the paired statistical tests (III-J), the analytical cost model (III-K), the four re-implemented
single-layer comparators (IV-H) and the bit-position-distribution study (IV-I), and regenerates all tables
(3–8, 10, 11) and figures (2–8) from the raw per-seed outputs.

```
sgcl/common.py        settings (Table 2), data splits, ResNet-18 / EfficientNet-B0 (32x32), training loop
                      (SGD, cosine, curriculum FAT, residual regularisation, SAM), IEEE-754 single-bit fault
                      injector (weights + output activations, Bernoulli lambda, uniform / weighted bit positions),
                      Monte Carlo driver (paired seeds 0..N-1), analytical cost model
sgcl/hardening.py     layer-wise sensitivity S_k, critical set (T = 2.5 pp), activation-range restriction
                      (1.1 x 99.99th percentile over 5 000 training images, logits +/-20), duplicate-and-compare
sgcl/optimize.py      sensitivity-aware iterative magnitude pruning (0.45 / 0.25 / 0 by S_k, 3 steps, fine-tune),
                      FP16 / FP8-E4M3 (per-channel scale) layer wrappers, policies MP-1..MP-5, cost of a configuration
sgcl/comparators.py   range restriction [34], FAT-RABBIT [35] (constant lambda + SAM), RAP [40], RL bit protection [36]
sgcl/stats.py         paired two-sided z-test on per-seed CER, 95 % CI, Bonferroni m = 4 (adjusted p vs 0.05)
run_pipeline.py       end-to-end driver with per-stage caching (safe to interrupt and restart)
make_report.py        tables (csv + markdown), figures (png + pdf), per-seed CSVs, summary.json
tests_smoke.py        unit checks of the fault model and the statistics (seconds, no training)
colab_run.ipynb       one-click Google Colab notebook (results cached on Google Drive)
```

## Quick start

```bash
pip install -r requirements.txt
python tests_smoke.py                                   # fault model + statistics checks (~10 s)
python run_pipeline.py --budget colab --out results --data data      # all experiments (GPU)
python make_report.py --results results --out report                 # tables, figures, per-seed CSVs
```

`run_pipeline.py` options:

| option | meaning |
|---|---|
| `--budget paper` | exact protocol of Table 2: 200 epochs, 30 fine-tuning epochs per pruning step, full test split, N = 500 |
| `--budget colab` | same structure on a free-GPU budget: 30 epochs, 5 fine-tuning epochs, fixed stratified 2 000-image test subset per trial, N = 500 (secondary λ / sparsity sweeps: 250) |
| `--budget smoke` | minutes-long CPU check of every code path (meaningless numbers) |
| `--pairs` | architecture:dataset pairs, default `resnet18:cifar10,efficientnet_b0:cifar10,resnet18:gtsrb,efficientnet_b0:gtsrb` |
| `--stages` | any subset of `baseline,vuln,sens,arch,fat,opt,unified,comparators,bitdist` |
| `--extra-archs vgg16,mobilenet_v2` | additional CIFAR-10 FAT rows for Table 5 |
| `--lambdas 1e-8,...,1e-4` | λ sweep for the baseline and FAT curves (Fig. 2, Fig. 5); the operating point and its ×10 / ÷10 neighbours use N trials, the rest N2 |
| `--lam-op 1e-5` | operating point λ for every other stage (Table 2 value by default); `--lam-fat-max` defaults to 2 × λ_op |
| `--fat-weights 0` | inject activations only during fault-aware training (default 1: weights and activations, as at evaluation) |
| `--n-trials`, `--epochs`, `--eval-subset` | override individual settings |

Every stage writes `results/<arch>_<dataset>/<config>.json` (per-seed accuracy and CER for seeds 0..N-1,
means, variances, standard errors, costs, settings) and is skipped on re-run if the file exists; checkpoints are in
`results/ckpt/`. The whole run can therefore be split across several Colab sessions.

Stages per pair:

| stage | what is produced | manuscript |
|---|---|---|
| baseline | training, clean test accuracy | — |
| vuln | N paired trials at λ = 10⁻⁶, 10⁻⁵, 10⁻⁴ (+ 3·10⁻⁶, 3·10⁻⁵) | Table 3, Fig. 2 |
| sens | S_k per block at λ = 10⁻⁵, critical set C = {k : S_k ≥ 2.5 pp} | Fig. 3 |
| arch | clip-only, duplicate-and-compare-only, β-regularised retraining, all three together | Table 4, Fig. 4 |
| fat | curriculum FAT (λ_e = λ_max·e/E, λ_max = 2·10⁻⁵), λ sweep | Table 5, Fig. 5 |
| opt | MP-1…MP-5 on the baseline; pruning sweep (Table-2 ratios × 1/3 … 5/3) with/without MP-3 | Table 6, Fig. 6 |
| unified | β + FAT training → sensitivity-aware pruning under FAT → MP-3 + clipping + duplicate-and-compare; paired tests vs baseline | Tables 7–8, Figs. 7–8 |
| comparators | range restriction, FAT-RABBIT, RAP, RL bit protection on the same checkpoint / seeds; paired tests vs unified | Table 10 |
| bitdist | exponent- and mantissa-weighted bit-position distributions for baseline and unified | Table 11 |

`opt`, `unified`, `comparators` and `bitdist` run for ResNet-18 (CIFAR-10 and GTSRB; comparators and bitdist on
CIFAR-10 only), as in the manuscript.

## Fault model

Each element of every weight tensor and every output-activation tensor of the convolutional and linear layers is
corrupted independently with probability λ by flipping one bit of its IEEE-754 representation (32-bit for FP32 layers,
16-bit / 8-bit words for FP16 / FP8 layers). The bit position is uniform by default; the exponent-weighted
(0.05 / 0.60 / 0.35) and mantissa-weighted (0.05 / 0.10 / 0.85) distributions of Section IV-I are selected with
`dist=`. Trial *i* uses `torch.Generator().manual_seed(i)`, so seed *i* draws the same fault sequence in every
configuration (paired design). Weight faults are restored after each trial. Batch-normalisation tensors and residual
additions are not injected but propagate faults, as described in Section III-C.

## Fault-aware training details (Algorithm 2)

Each training step at λ_e > 0 runs a clean forward pass (which updates the batch-normalisation statistics) and a
faulty forward pass with the same injector used at evaluation; the faulty pass is back-propagated while the faults
are applied (straight-through at the injected elements), batch-normalisation uses its running statistics in that
pass so that one overflowed image cannot corrupt the whole batch, and corrupted activations that are non-finite or
exceed 10⁴ × the median clean activation of the tensor are replaced by zero (NaN/overflow filtering, as in DieHardNet).
Samples whose loss still overflows are excluded, gradients are clipped to norm 5, and the numbers of filtered
elements, excluded samples and skipped steps are logged per epoch in `train_*.json`. Evaluation never filters.

## Notes on reproducibility

* All randomness is seeded (`train_seed`, Monte Carlo seeds 0..N-1, split permutation). GPU convolution kernels are
  not bit-wise deterministic across hardware, so re-running training on a different GPU gives a statistically
  equivalent but not bit-identical checkpoint; re-running the fault-injection stages on the *same* checkpoint gives
  identical per-seed outputs.
* `settings.json` in the results folder records the budget and every parameter used.
* The comparators are faithful re-implementations of the published mechanisms at the granularity of this study
  (see the docstring of `sgcl/comparators.py`); they are not the authors' original code.

## License

MIT (see `LICENSE`). Cite the article when using this code (`CITATION.cff`).
