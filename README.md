# Moon Online Models

Moon Online Models (`moon-online-models`) is a MoonBit library for online regression and incremental learning. It is designed for real-time model updating and continuous learning scenarios such as CTR (Click-Through Rate) prediction, device metrics monitoring, and real-time business forecasting.

## Architecture & Core Features

- **Recursive Least Squares (RLS):** Online linear regression with a forgetting factor, suitable for time-series and non-stationary data streams.
- **Online Logistic Regression (FTRL-Proximal):** Follow-The-Regularized-Leader proximal algorithm with L1 and L2 regularization support. Highly optimized for sparsity, ideal for CTR prediction.
- **Feature Standardization:** Welford's online algorithm to incrementally track the mean and variance, standardizing data streams on the fly.
- **Model Snapshot (Serialization):** Built-in `ToJson` and `FromJson` serialization for all models using MoonBit's core JSON derive macros.
- **Incremental Evaluation:** Online tracking of metrics like LogLoss and Mean Squared Error (MSE).

## Getting Started

Add the dependency to your project:
```bash
moon add Lyllyl789/moon-online-models
```

### Example: FTRL Logistic Regression

```mbt nocheck
test "ftrl example" {
  // Initialize FTRL model with 2 features
  let model = FTRL::new(2, alpha=0.1, l1=0.5, l2=1.0)
  
  // Update model incrementally with new data stream
  model.update([1.0, 0.0], 1.0)
  model.update([0.0, 1.0], 0.0)
  
  // Predict on new data
  let pred = model.predict([1.0, 0.0])
  inspect(pred > 0.5, content="true")
}
```

### Example: Tracking Incremental Metrics

```mbt nocheck
test "metrics example" {
  let metrics = MetricsTracker::new()
  metrics.update(0.9, 1.0) // predict 0.9, actual 1.0
  metrics.update(0.2, 0.0) // predict 0.2, actual 0.0
  
  let logloss = metrics.log_loss()
  let mse = metrics.mse()
  inspect(mse < 0.1, content="true")
}
```

## Hackathon / OSC 2026 Declaration

This project is natively developed in MoonBit from scratch.
- **License**: Apache-2.0
- **Scope**: Implements machine learning models natively in MoonBit, addressing the lack of online learning libraries in the MoonBit ecosystem.
- **Extensibility**: Future updates can easily extend the `OnlineLearner` trait to support sparse features, adaptive learning rates (Adagrad), and more.
