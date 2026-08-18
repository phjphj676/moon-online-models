# moon-online-models

`moon-online-models` is a native MoonBit toolkit for machine learning on
streaming data. It provides online models, incremental feature processing,
evaluation metrics, monitoring utilities, and deployment primitives for use
cases such as CTR prediction, risk scoring, device telemetry, ranking, and
real-time forecasting.

The library is designed for event-by-event updates, bounded state, explicit
validation, deterministic experiments, and compilation to MoonBit-supported
targets.

## Core capabilities

### Online learning

- RLS, SGD/FTRL logistic regression, Adagrad, ridge, Huber, and quantile
  regression.
- Softmax regression, Gaussian and Bernoulli Naive Bayes, online k-means,
  kernel models, PCA, factorization, tree stumps, ensembles, and time-series
  models.

### Streaming data and features

- Dense and sparse vectors with dimension-safe operations.
- Feature hashing, categorical encoding, CSV parsing, feature crosses, and
  polynomial features.
- Online joins, rolling windows, reservoirs, stratified sampling, bootstrap
  counters, schema validation, and bounded feature storage.

### Evaluation and reliability

- Exact and histogram AUC, calibration, regression, ranking, CTR, and
  confusion-matrix metrics.
- Conformal intervals, drift detection, fairness diagnostics,
  cost-sensitive thresholds, gradient guards, and prediction guards.
- Snapshots with checksums, model registries, deployment state transitions,
  canary experiments, serving batchers, alerts, error budgets, audit trails,
  data lineage, privacy budgets, and reproducibility manifests.

## Quick start

Add the package to a MoonBit project:

```bash
moon add phjphj676/moon-online-models
```

Create and update a dense online classifier:

```mbt nocheck
let model = @moon-online-models.AdagradLogisticRegression::new(
  3,
  learning_rate=0.1,
)
model.update([1.0, 0.0, 0.2], 1.0)
model.update([0.0, 1.0, -0.2], 0.0)
let probability = model.predict([1.0, 0.0, 0.2])
```

Use sparse hashed features for high-cardinality event data:

```mbt nocheck
let hasher = @moon-online-models.FeatureHasher::new(1_000_000)
let features = hasher.encode(["country=CN", "device=mobile", "slot=home"])
let model = @moon-online-models.SparseAdagradClassifier::new(1_000_000)
model.update(features, 1.0)
let probability = model.predict(features)
```

## CLI and reproducible commands

The repository includes a native benchmark command under `cmd/benchmark` and
a PowerShell wrapper under `benchmarks/run.ps1`.

```bash
moon version --all
moon update
moon check --target all
moon test --target all
moon run --target native cmd/benchmark
```

On Windows, the wrapper records the benchmark wall-clock time:

```powershell
.\benchmarks\run.ps1
```

Formatting and generated-interface checks:

```bash
moon fmt && git diff --exit-code
moon info && git diff --exit-code
```

## Architecture

```text
moon-online-models/
├── *.mbt                       Core models and streaming primitives
├── extended_models_test.mbt    Boundary and behavior coverage
├── cmd/benchmark/              Native reproducible benchmark command
├── benchmarks/                 Runner, methodology, and measured results
└── .github/workflows/          Cross-platform checks and package publishing
```

The root package is dependency-light and uses MoonBit core packages for math
and JSON support. Algorithms expose explicit counters, reset operations, and
dimension-safe boundaries so they can be embedded in long-running streams.

## Benchmark

The benchmark trains an `AdagradLogisticRegression` model on 20,000
deterministic events and reports update count, exact AUC, final weight norm,
and a stable weight checksum. The latest measured reference is maintained in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md); elapsed time is hardware- and
runtime-dependent.

## Tests

The black-box suite covers empty inputs, dimension mismatches, out-of-range
indices, non-finite values, sparse features, CSV quoting, metric degeneracy,
bounded capacities, drift detection, feature storage, serving, deployment,
audit logging, and cost-sensitive decisions.

Run all configured targets with:

```bash
moon test --target all
```

## CI

GitHub Actions installs the current stable MoonBit toolchain and checks
Ubuntu, macOS, and Windows. Each job runs `moon version --all`, `moon update`,
`moon check --target all`, `moon test --target all`, `moon fmt`, and `moon info`.
The publishing workflow is manual and runs the same prepublish checks before
calling `moon publish`.

## Package and license

- Package: `phjphj676/moon-online-models@0.2.0`
- GitHub: <https://github.com/phjphj676/moon-online-models>
- GitLink: <https://www.gitlink.org.cn/phjphj676/moon-online-models>
- License: Apache-2.0; see [`LICENSE`](LICENSE).
