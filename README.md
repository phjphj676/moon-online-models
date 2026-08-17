# moon-online-models

`phjphj676/moon-online-models` is a native MoonBit toolkit for applications
that learn from events continuously: CTR/risk scoring, device telemetry,
incremental forecasting, ranking, feature streams, and model monitoring. It
is the acceptance version of the August 2026 MoonBit Hackathon project.

## What is included

- Dense online learners: RLS, SGD/FTRL logistic regression, Adagrad, ridge,
  Huber, quantile, softmax, Naive Bayes, k-means, kernels, PCA, factorization,
  tree stumps, ensembles, and time-series models.
- Sparse and streaming data: sparse vectors, hashing, CSV parsing, categorical
  encoding, feature crosses, online joins, reservoirs, bootstrapping, and
  schema-checked feature storage.
- Evaluation and safety: exact and histogram AUC, calibration, regression and
  ranking metrics, conformal intervals, drift detectors, fairness gaps,
  cost-sensitive thresholds, validation, gradient/prediction guards.
- Operations: model snapshots and checksums, registry and deployment state
  transitions, canary experiments, serving batchers, SLO/error budgets,
  alerts, audit trails, data lineage, privacy budgets, and reproducibility
  manifests.

The implementation is deliberately dependency-light: the root package only
uses `moonbitlang/core/math` and `moonbitlang/core/json`. Public state is
bounded where a stream can grow without limit; callers can inspect counters,
reset state, and reject malformed dimensions at the boundary.

## Install

```bash
moon add phjphj676/moon-online-models
```

## Minimal example

```mbt nocheck
let model = @moon-online-models.AdagradLogisticRegression::new(
  3,
  learning_rate=0.1,
)
model.update([1.0, 0.0, 0.2], 1.0)
model.update([0.0, 1.0, -0.2], 0.0)
let probability = model.predict([1.0, 0.0, 0.2])
```

For sparse CTR-style features:

```mbt nocheck
let hasher = @moon-online-models.FeatureHasher::new(1_000_000)
let features = hasher.encode(["country=CN", "device=mobile", "slot=home"])
let model = @moon-online-models.SparseAdagradClassifier::new(1_000_000)
model.update(features, 1.0)
let probability = model.predict(features)
```

## Verification

Run the same checks locally that the repository CI runs:

```bash
moon version --all
moon update
moon check --target all
moon test --target all
moon fmt && git diff --exit-code
moon info && git diff --exit-code
```

The local wasm-gc test suite contains 16 boundary and behavior scenarios;
the native benchmark trains 20,000 deterministic events and records its
measured output in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). Re-run
it with `benchmarks/run.ps1` on the acceptance machine.

## Package and repository

- Package: `phjphj676/moon-online-models@0.2.0`
- GitHub: <https://github.com/phjphj676/moon-online-models>
- GitLink mirror: <https://www.gitlink.org.cn/phjphj676/moon-online-models>
- Default branch: `master`
- License: Apache-2.0; see [`LICENSE`](LICENSE).
- The project proposal and acceptance notes are retained in [`申报书.md`](申报书.md).

The code is original MoonBit implementation authored by the repository owner.
No generated build directory or credentials are part of the package.
