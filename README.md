# AetherNet Research Lab 0.2.0

A local, zero-paid-API model laboratory for comparing different learning mechanisms. It is designed to run on an ordinary Windows laptop using Python and scikit-learn. It does **not** need a GPU or cloud inference service for these small experiments; the work uses the host computer's physical resources, normally its CPU.

## What's new in 0.2.0

- A local-only browser dashboard with live benchmark controls, result table, JSON export, and CSV export.
- Model comparisons: MLP, Extreme Learning Machine-style fixed random features, hyperdimensional prototype classifier, RBF network, KNN, nearest centroid, random forest, the original AetherNet gated KNN hybrid, AetherGrow (validation-selected hidden width), and AetherCascade.
- Per-model held-out accuracy, balanced accuracy, macro-F1, fit time, prediction time, serialized model size, and expert routing rate where applicable.
- Tests for the new architecture implementations and benchmark wiring.

## Run on Windows

1. Extract the ZIP to a folder such as `C:\Users\Jagan\Downloads\aethernet_lab_v02`.
2. Open PowerShell in that folder (the folder must contain `index.html`, `server.py`, `architectures.py` and `requirements.txt`).
3. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe server.py
```

4. Open `http://127.0.0.1:8010` in your browser. Keep the terminal open while using the app. Press `Ctrl+C` to stop the local server.

The app binds to `127.0.0.1` only. It does not expose the user's home directory; the web server serves this project folder. The browser dashboard sends benchmark requests only to the local server.

### Run benchmark from the terminal

```powershell
.\.venv\Scripts\python.exe benchmarks_v2.py --repeats 3 --samples 900
```

For a quicker run:

```powershell
.\.venv\Scripts\python.exe benchmarks_v2.py --repeats 1 --samples 500 --dataset moons
```

Results are saved in `results_v2.json` in the project folder. All datasets are generated locally; no benchmark data is uploaded.

## Candidate architecture: AetherGrow

AetherGrow evaluates a predefined set of hidden widths for an ELM-style fixed-random-feature model on a calibration split, then selects the smallest width within an accuracy tolerance of the best calibration score. It refits that selected width on all training rows. This is a model-selection experiment layered on established ELM methods; it is not claimed as a new family.

## Candidate architecture: AetherCascade

AetherCascade routes each query through:

1. a small MLP as the initial predictor;
2. an Extreme Learning Machine-style expert for low-confidence inputs;
3. KNN as a last fallback for inputs still marked uncertain.

The routing thresholds are selected using a calibration split of the training data, not test labels. The routing penalty is a design proxy for extra calls—not a measurement of power, energy, or FLOPs. The experiment may fail to outperform simpler baselines, and that negative outcome must be retained.

## Interpret metrics carefully

- **Accuracy / balanced accuracy / macro-F1** measure predictions on held-out test data.
- **Fit/predict seconds** are wall-clock measurements on this machine and vary with hardware, background activity and dependency versions.
- **Serialized size** estimates storage size of a saved model object. It is not peak RAM usage.
- **Routing-call rates** record how often a specialist was invoked. They do not directly prove lower energy usage.
- **Synthetic datasets** are useful for deterministic smoke tests, but they do not prove performance on real-world data.

## Is it a new neural network?

Not established. MLPs, ELMs, hyperdimensional computing, RBF networks, nearest-neighbor methods, random forests and cascaded/adaptive inference all have prior art. AetherCascade is a candidate composition and routing experiment, not a verified first-ever invention. A meaningful original contribution would require a well-defined new mechanism plus reproducible gains over strong baselines on multiple public real-world datasets.

## Safety and privacy

The benchmark uses locally generated synthetic data. Do not upload private data to public repositories. The server is intentionally restricted to loopback; do not change its bind address unless you understand the security implications.
