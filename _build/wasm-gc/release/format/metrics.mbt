///|
/// Incremental metrics tracker.
pub struct MetricsTracker {
  mut count : Double
  mut sum_squared_error : Double
  mut sum_log_loss : Double
}

///|
/// Create a new metrics tracker.
pub fn MetricsTracker::new() -> MetricsTracker {
  { count: 0.0, sum_squared_error: 0.0, sum_log_loss: 0.0 }
}

///|
/// Update metrics with a new prediction and ground truth label.
pub fn MetricsTracker::update(
  self : MetricsTracker,
  pred : Double,
  label : Double,
) -> Unit {
  self.count += 1.0
  let err = pred - label
  self.sum_squared_error += err * err

  // LogLoss: - (y * log(p) + (1 - y) * log(1 - p))
  // Bound pred to avoid log(0)
  let eps = 1.0e-15
  let p = if pred < eps {
    eps
  } else if pred > 1.0 - eps {
    1.0 - eps
  } else {
    pred
  }

  let log_loss = if label > 0.5 { -@math.ln(p) } else { -@math.ln(1.0 - p) }
  self.sum_log_loss += log_loss
}

///|
/// Get the current Mean Squared Error (MSE).
pub fn MetricsTracker::mse(self : MetricsTracker) -> Double {
  if self.count == 0.0 {
    0.0
  } else {
    self.sum_squared_error / self.count
  }
}

///|
/// Get the current LogLoss.
pub fn MetricsTracker::log_loss(self : MetricsTracker) -> Double {
  if self.count == 0.0 {
    0.0
  } else {
    self.sum_log_loss / self.count
  }
}
