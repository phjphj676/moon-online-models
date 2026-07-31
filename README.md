# Moon Online Models

Moon Online Models (`moon-online-models`) is a MoonBit library for online regression and incremental learning. It is designed for real-time model updating and continuous learning scenarios such as CTR (Click-Through Rate) prediction, device metrics monitoring, and real-time business forecasting.

## Features

- **Recursive Least Squares (RLS):** Online linear regression with forgetting factor.
- **Online Logistic Regression:** FTRL-Proximal algorithm with L1/L2 regularization for sparse and dense features.
- **Feature Standardization:** Incremental tracking of mean and variance to standardize incoming streams of data.
- **Model Snapshot:** Serialize and deserialize model weights for persistence.
- **Incremental Evaluation:** Online tracking of metrics like LogLoss, RMSE, etc.

## Quick Start

Add the dependency to your project:
```bash
moon add Lyllyl789/moon-online-models
```
