///|
/// Streaming ridge regression with a diagonal preconditioner.
pub struct OnlineRidgeRegression {
  weights : Array[Double]
  diagonal : Array[Double]
  learning_rate : Double
  l2 : Double
  mut intercept : Double
  mut intercept_diagonal : Double
  mut steps : Int
}

///|
pub fn OnlineRidgeRegression::new(
  dimension : Int,
  learning_rate? : Double = 0.01,
  l2? : Double = 1.0e-4,
) -> OnlineRidgeRegression {
  let size = if dimension < 0 { 0 } else { dimension }
  {
    weights: Array::make(size, 0.0),
    diagonal: Array::make(size, 1.0),
    learning_rate,
    l2,
    intercept: 0.0,
    intercept_diagonal: 1.0,
    steps: 0,
  }
}

///|
pub fn OnlineRidgeRegression::dimension(self : OnlineRidgeRegression) -> Int {
  self.weights.length()
}

///|
pub fn OnlineRidgeRegression::weights(
  self : OnlineRidgeRegression,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn OnlineRidgeRegression::intercept(self : OnlineRidgeRegression) -> Double {
  self.intercept
}

///|
pub fn OnlineRidgeRegression::steps(self : OnlineRidgeRegression) -> Int {
  self.steps
}

///|
pub fn OnlineRidgeRegression::predict(
  self : OnlineRidgeRegression,
  features : Array[Double],
) -> Double {
  self.intercept + dot_product(self.weights, features)
}

///|
pub fn OnlineRidgeRegression::update(
  self : OnlineRidgeRegression,
  features : Array[Double],
  label : Double,
) -> Unit {
  self.update_weighted(features, label, 1.0)
}

///|
pub fn OnlineRidgeRegression::update_weighted(
  self : OnlineRidgeRegression,
  features : Array[Double],
  label : Double,
  sample_weight : Double,
) -> Unit {
  let error = (self.predict(features) - label) * sample_weight
  let limit = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  for i in 0..<limit {
    let gradient = error * features[i] + self.l2 * self.weights[i]
    self.diagonal[i] += features[i] * features[i] * sample_weight
    self.weights[i] -= self.learning_rate * gradient / self.diagonal[i].sqrt()
  }
  self.intercept_diagonal += sample_weight
  self.intercept -= self.learning_rate * error / self.intercept_diagonal.sqrt()
  self.steps += 1
}

///|
pub fn OnlineRidgeRegression::loss(
  self : OnlineRidgeRegression,
  features : Array[Double],
  label : Double,
) -> Double {
  0.5 * squared_error(self.predict(features), label) +
  0.5 * self.l2 * squared_norm(self.weights)
}

///|
pub fn OnlineRidgeRegression::residual(
  self : OnlineRidgeRegression,
  features : Array[Double],
  label : Double,
) -> Double {
  label - self.predict(features)
}

///|
pub fn OnlineRidgeRegression::rmse(
  self : OnlineRidgeRegression,
  samples : Array[Array[Double]],
  labels : Array[Double],
) -> Double {
  let size = if samples.length() < labels.length() {
    samples.length()
  } else {
    labels.length()
  }
  if size == 0 {
    0.0
  } else {
    let mut total = 0.0
    for i in 0..<size {
      total += squared_error(self.predict(samples[i]), labels[i])
    }
    (total / size.to_double()).sqrt()
  }
}

///|
pub fn OnlineRidgeRegression::reset(self : OnlineRidgeRegression) -> Unit {
  self.weights.fill(0.0)
  self.diagonal.fill(1.0)
  self.intercept = 0.0
  self.intercept_diagonal = 1.0
  self.steps = 0
}

///|
/// Robust online regression using the Huber loss.
pub struct OnlineHuberRegression {
  weights : Array[Double]
  learning_rate : Double
  delta : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn OnlineHuberRegression::new(
  dimension : Int,
  learning_rate? : Double = 0.01,
  delta? : Double = 1.0,
  l2? : Double = 0.0,
) -> OnlineHuberRegression {
  {
    weights: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    learning_rate,
    delta: if delta <= 0.0 {
      1.0
    } else {
      delta
    },
    l2,
    steps: 0,
  }
}

///|
pub fn OnlineHuberRegression::predict(
  self : OnlineHuberRegression,
  features : Array[Double],
) -> Double {
  dot_product(self.weights, features)
}

///|
pub fn OnlineHuberRegression::weights(
  self : OnlineHuberRegression,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn OnlineHuberRegression::update(
  self : OnlineHuberRegression,
  features : Array[Double],
  label : Double,
) -> Unit {
  let residual = self.predict(features) - label
  let magnitude = if residual < 0.0 { -residual } else { residual }
  let gradient_residual = if magnitude <= self.delta {
    residual
  } else if residual < 0.0 {
    -self.delta
  } else {
    self.delta
  }
  let limit = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  for i in 0..<limit {
    let gradient = gradient_residual * features[i] + self.l2 * self.weights[i]
    self.weights[i] -= self.learning_rate * gradient
  }
  self.steps += 1
}

///|
pub fn OnlineHuberRegression::loss(
  self : OnlineHuberRegression,
  features : Array[Double],
  label : Double,
) -> Double {
  smooth_l1_loss(self.predict(features) - label, beta=self.delta) +
  0.5 * self.l2 * squared_norm(self.weights)
}

///|
pub fn OnlineHuberRegression::steps(self : OnlineHuberRegression) -> Int {
  self.steps
}

///|
/// Online quantile regression for prediction intervals and tail forecasting.
pub struct OnlineQuantileRegression {
  weights : Array[Double]
  learning_rate : Double
  quantile : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn OnlineQuantileRegression::new(
  dimension : Int,
  quantile? : Double = 0.5,
  learning_rate? : Double = 0.01,
  l2? : Double = 0.0,
) -> OnlineQuantileRegression {
  {
    weights: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    learning_rate,
    quantile: clamp(quantile, 1.0e-6, 1.0 - 1.0e-6),
    l2,
    steps: 0,
  }
}

///|
pub fn OnlineQuantileRegression::quantile(
  self : OnlineQuantileRegression,
) -> Double {
  self.quantile
}

///|
pub fn OnlineQuantileRegression::predict(
  self : OnlineQuantileRegression,
  features : Array[Double],
) -> Double {
  dot_product(self.weights, features)
}

///|
pub fn OnlineQuantileRegression::update(
  self : OnlineQuantileRegression,
  features : Array[Double],
  label : Double,
) -> Unit {
  let residual = label - self.predict(features)
  let gradient = if residual >= 0.0 {
    -self.quantile
  } else {
    1.0 - self.quantile
  }
  let limit = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  for i in 0..<limit {
    self.weights[i] -= self.learning_rate *
      (gradient * features[i] + self.l2 * self.weights[i])
  }
  self.steps += 1
}

///|
pub fn OnlineQuantileRegression::pinball_loss(
  self : OnlineQuantileRegression,
  features : Array[Double],
  label : Double,
) -> Double {
  let error = label - self.predict(features)
  if error >= 0.0 {
    self.quantile * error
  } else {
    (self.quantile - 1.0) * error
  }
}

///|
pub fn OnlineQuantileRegression::weights(
  self : OnlineQuantileRegression,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub struct RunningMean {
  mut count : Double
  mut mean : Double
}

///|
pub fn RunningMean::new() -> RunningMean {
  { count: 0.0, mean: 0.0 }
}

///|
pub fn RunningMean::update(self : RunningMean, value : Double) -> Unit {
  self.count += 1.0
  self.mean += (value - self.mean) / self.count
}

///|
pub fn RunningMean::merge(self : RunningMean, other : RunningMean) -> Unit {
  if other.count > 0.0 {
    let total = self.count + other.count
    self.mean = (self.mean * self.count + other.mean * other.count) / total
    self.count = total
  }
}

///|
pub fn RunningMean::value(self : RunningMean) -> Double {
  self.mean
}

///|
pub fn RunningMean::count(self : RunningMean) -> Double {
  self.count
}
