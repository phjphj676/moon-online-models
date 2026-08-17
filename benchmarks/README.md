# Reproducible benchmark

The benchmark trains `AdagradLogisticRegression` on 20,000 deterministic
events, scores the same events online, and prints a checksum of the final
weights. The workload is intentionally deterministic; elapsed time is the
only machine-dependent value.

From the repository root:

```powershell
moon run --target native cmd/benchmark
.\benchmarks\run.ps1
```

The output records samples, update steps, exact AUC, weight L1 norm, a stable
weight checksum, and wall-clock milliseconds. `RESULTS.md` contains the
maintainer's latest measured run and hardware/runtime context.
