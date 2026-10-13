# AetherNet Lab 0.2.0 — architecture landscape and hypotheses

## Established methods included as baselines

- **MLP:** trainable multilayer perceptron.
- **Extreme Learning Machine (ELM):** fixed random nonlinear hidden layer and trained output readout.
- **Hyperdimensional computing:** high-dimensional vector-symbolic representations; this prototype uses random projection encoding and class prototype bundling.
- **RBF network:** radial basis features around clustering-derived centers with a learned readout.
- **KNN / nearest centroid:** instance-based and prototype-based baselines.
- **Random forest:** non-neural tabular baseline.
- **Selective/adaptive inference:** route uncertain inputs to additional compute.

Representative repositories and starting points:

- Efficient KAN: https://github.com/Blealtan/efficient-kan
- TorchHD: https://github.com/hyperdimensional-computing/torchhd
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
- Adaptive computation reading list: https://github.com/koayon/awesome-adaptive-computation
- Online recurrent ELM example: https://github.com/chickenbestlover/Online-Recurrent-Extreme-Learning-Machine
- Local test-time learning paper: https://www.nature.com/articles/s41598-025-31491-3
- Fully test-time adaptation for tabular data (AAAI 2025): https://ojs.aaai.org/index.php/AAAI/article/view/34466

This repository-oriented scan is a starting point, not an exhaustive novelty search or patent review.

## Candidate: AetherGrow

**Hypothesis:** selecting the smallest hidden width whose calibration accuracy lies within a fixed tolerance of the best candidate can reduce stored weights and potentially inference time while preserving test accuracy. The implementation searches a fixed set of ELM-style widths and evaluates the selected width on a held-out test split. This is adaptive capacity selection around an established model family, not claimed first-ever novelty.

**Failure mode:** the calibration partition may select a width that does not generalize. Report the entire width-versus-calibration curve and validate on independent data.

## Candidate: AetherCascade

**Hypothesis:** a small MLP can handle easy predictions, an ELM-style model can refine some uncertain predictions, and KNN can serve as a last fallback. A calibration-tuned router may retain useful accuracy while reducing full use of more expensive specialists.

**Procedure:**

1. Split training data into core-fit and calibration partitions.
2. Fit MLP, ELM and KNN only on core-fit data.
3. Compare routing thresholds on calibration data with the objective `validation_accuracy - lambda * weighted_route_rate`.
4. Freeze thresholds before evaluating on a held-out test partition.
5. Measure predictive scores, wall-clock fit and prediction time, serialized model storage and route frequency.

The current route penalty uses hand-chosen weights (ELM call = 0.5 units; KNN call = 1 unit) for an experiment. It is not an empirically measured cost model. A follow-up should replace it with a cost estimate measured on the target hardware and test whether the policy transfers across tasks.

## Research plan

1. Run a reproducible multi-seed benchmark on synthetic datasets.
2. Add strong baselines and real public tabular datasets.
3. Use nested validation or a separate calibration set to prevent test leakage.
4. Compare at matched latency/cost budgets, not accuracy alone.
5. Perform ablations: remove ELM, remove KNN, change gates, and vary the routing penalty.
6. Publish all results, including runs where the candidate loses.

## Novelty boundary

This code does **not** prove a new neural-network family, global novelty, lower energy use, or general intelligence. Cascaded models and adaptive computation are established ideas. The potential research contribution, if any, lies in a specific routing/learning rule whose performance can be independently replicated and shown to improve a practical metric.
