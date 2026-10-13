# AetherNet: practical user value, limitations, and scalability

## What a user gets today

AetherNet is a local research bench, not a general-purpose chatbot or a claim to replace frontier AI. It lets a learner or developer run the same small classification task through multiple model families and inspect held-out accuracy, balanced accuracy, macro-F1, fit time, prediction time, serialized model size, and selective-expert routing rates. CSV/JSON exports make results easier to reproduce and discuss.

## Who may find it useful

- **Students and independent researchers:** learn what changes when the representation or learning rule changes without buying API credits or renting a GPU.
- **Developers with low budgets:** compare CPU-friendly methods and test whether a smaller model is adequate for a specific task.
- **Privacy-conscious users:** keep benchmark execution on the local machine; the server binds to loopback and does not upload benchmark data.
- **Research collaborators:** share code, protocol, and negative results so others can reproduce experiments.

## What it is not yet useful for

- It is not a replacement for a language model and cannot perform open-ended writing, complex reasoning, or broad world knowledge from these tabular classifiers.
- Its results do not prove general intelligence, universal superiority, no hallucinations, or zero physical computation.
- The current candidate mechanisms (AetherGrow and AetherCascade) are based on established ELM/model-selection and cascaded-inference ideas. Novelty is unproven.
- Current timing and serialized size are proxies. Serialized size is not peak RAM, and routing rate is not joules or FLOPs.

## Benchmarks and data

The four synthetic tasks are deterministic generated data useful for smoke tests and controlled comparisons. Version 0.3 adds scikit-learn's bundled Breast Cancer Wisconsin Diagnostic, Wine, and Digits datasets. These datasets are loaded locally from the installed scikit-learn package; no runtime dataset download is needed. The Wine dataset has only 178 records, so the UI sample setting is a maximum, not a promise that every dataset has that many records. Source documentation: https://scikit-learn.org/stable/datasets.html

Each repeat uses a stratified train/test split. Standardization is fitted on the training split only. The adaptive-width and cascade mechanisms choose settings using calibration data inside the training split, not test labels. Synthetic and bundled datasets remain small benchmarks; follow-up work should use more public datasets and repeated/nested validation before broad claims.

## Benefits and trade-offs

| Benefit | Trade-off / limitation |
|---|---|
| No paid AI API required | Uses the user's actual CPU/RAM and electricity |
| Runs on an ordinary laptop for small datasets | Larger datasets or heavier model families can be slow or memory-intensive |
| Local-only dashboard and data | Local-only also means no built-in team sync or hosted results |
| Multiple methods in one repeatable interface | Fair comparison still depends on good splits and sensible hyperparameter budgets |
| Exports make results shareable | Users must avoid exporting private source data and interpret metrics correctly |
| Candidate routes may skip some specialist calls | Fewer calls do not automatically mean lower latency or energy; routing has overhead |

## Scalability

The UI and benchmark API are suitable for an individual local experiment. Scaling to larger datasets means controlling memory (especially pairwise methods such as KNN), setting benchmark time budgets, using stratified cross-validation, and reporting confidence intervals. Supporting many collaborators would require access controls, result provenance, versioned environments, and security/privacy review. Distributed or cloud execution is deliberately outside the zero-cost local prototype.

## Originality process

Before claiming a new architecture, describe the mechanism mathematically, perform a focused literature and repository search, compare with strong baselines, run ablations, use independent real datasets, report failures, and seek independent replication. Adaptive-width neural networks are already an active research area; see https://github.com/nec-research/Adaptive-Width-Neural-Networks. The current AetherGrow is much simpler and is not claimed to be novel.

## What would make it worth using?

A user-facing win should be concrete: for example, the same quality with a smaller saved model, lower measured latency on the user's hardware, or more reliable predictions on a defined task. The project should not optimize a single synthetic accuracy score. It should publish results per dataset and retain negative findings.
