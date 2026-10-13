# AetherNet Research Lab 0.2.0 — initial experiment record

## Experiment setup

- Four locally generated synthetic classification datasets: moons, circles, mixed tabular, and blobs.
- 900 examples per dataset; 80% train / 20% held-out test.
- Three fixed-seed repeats per dataset.
- Ten methods: MLP, ELM, hyperdimensional prototype, RBF network, KNN, nearest centroid, random forest, AetherNet gated KNN, AetherGrow, and AetherCascade.
- AetherGrow selects a hidden width by validation accuracy, choosing the smallest width within 1 percentage point of the best calibration score. It then refits the selected width on all training rows.
- AetherCascade thresholds are tuned on an internal calibration split; test labels are not used for gate selection.
- Full per-trial values are in `results_v2.json`.

## Selected outcomes

Accuracy is mean ± standard deviation across three runs, in percentage points. Storage is serialized model size; timings are from the environment that generated this package and are **not representative measurements of the user's laptop**.

| Dataset | AetherGrow | Fixed-width ELM (192 units) | MLP | Best observed baseline |
|---|---:|---:|---:|---|
| Moons | 94.44% ± 2.08 pp | 95.00% ± 1.36 pp | 89.63% ± 3.02 pp | ELM, 95.00% |
| Circles | 90.93% ± 1.46 pp | 91.11% ± 1.64 pp | 90.37% ± 0.69 pp | KNN, 91.48% |
| Mixed tabular | 82.78% ± 3.14 pp | 82.22% ± 6.29 pp | 85.00% ± 3.18 pp | MLP / random forest, 85.00% |
| Blobs | 100.00% ± 0.00 pp | 100.00% ± 0.00 pp | 99.07% ± 0.69 pp | Several models reached 100% |

AetherGrow selected widths of 32–64 on moons, 32 on circles, 32–128 on mixed tabular, and 32 on blobs in these runs. Its average serialized size was approximately 2.5 KB, 2.1 KB, 11.1 KB, and 3.8 KB on the four tasks respectively, versus 8.2 KB, 8.2 KB, 23.5 KB, and 17.4 KB for the fixed 192-unit ELM under the same serialization method.

## Interpretation

- **Promising but limited:** AetherGrow retained near-ELM accuracy on moons and circles with a smaller stored hidden layer. On blobs it matched the ELM result with fewer stored features. These generated tasks are deliberately small and do not establish generalization to real data.
- **Not a universal win:** on mixed tabular, AetherGrow reached 82.78%, while MLP and random forest reached 85.00% in this three-run sample.
- **The cascade is not currently justified as an efficiency win:** its training procedure is slower, and its test accuracy did not consistently beat simpler methods. Its weighted routing objective uses hand-set penalties, not measured joules or FLOPs.
- **Hyperdimensional prototype struggled on the circles task** (about 51.5% mean accuracy). That is a useful negative result and a cue to revise the encoding if we pursue it further.
- Small sample counts and three seeds mean differences may be noise. No statistical-significance or first-ever claims are made.

## Next experiments

1. Compare AetherGrow and fixed-width ELM on real open tabular datasets with nested validation.
2. Increase repeats and report confidence intervals; freeze test sets before any architecture tuning.
3. Measure peak RAM and latency on the actual Windows laptop.
4. Compare at matched accuracy and matched latency/storage budgets, not accuracy alone.
5. Test ablations for the width set, calibration tolerance, activation function and ridge penalty.
6. Review prior work on ELM model selection, adaptive-width networks and model compression before making any novelty claim.

AetherGrow is a compact-width selection experiment around an established ELM-style method. AetherCascade composes established classifiers with a learned routing policy. Neither is proven to be a new model family; the candidate would need a reproducible advantage over strong baselines on real datasets to justify further research.
