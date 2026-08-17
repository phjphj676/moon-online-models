///|
/// Online binary classifier trained with Adagrad and optional elastic-net
/// regularization. It is a practical dense counterpart to FTRL for streams
/// where feature values are dense but their scales change over time.
pub struct AdagradLogisticRegression {
  weights : Array[Double]
  accumulator : Array[Double]
  learning_rate : Double
  epsilon : Double
  l1 : Double
  l2 : Double
  mut steps : Int
  mut seen : Double
}

///|
pub fn AdagradLogisticRegression::new(
  dimension : Int,
  learning_rate? : Double = 0.05,
  epsilon? : Double = 1.0e-8,
  l1? : Double = 0.0,
  l2? : Double = 0.0,
) -> AdagradLogisticRegression {
  let size = if dimension < 0 { 0 } else { dimension }
  {
    weights: Array::make(size, 0.0),
    accumulator: Array::make(size, 0.0),
    learning_rate,
    epsilon,
    l1,
    l2,
    steps: 0,
    seen: 0.0,
  }
}

///|
pub fn AdagradLogisticRegression::dimension(
  self : AdagradLogisticRegression,
) -> Int {
  self.weights.length()
}

///|
pub fn AdagradLogisticRegression::steps(
  self : AdagradLogisticRegression,
) -> Int {
  self.steps
}

///|
pub fn AdagradLogisticRegression::seen(
  self : AdagradLogisticRegression,
) -> Double {
  self.seen
}

///|
pub fn AdagradLogisticRegression::weights(
  self : AdagradLogisticRegression,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn AdagradLogisticRegression::accumulator(
  self : AdagradLogisticRegression,
) -> Array[Double] {
  copy_vector(self.accumulator)
}

///|
pub fn AdagradLogisticRegression::logit(
  self : AdagradLogisticRegression,
  features : Array[Double],
) -> Double {
  dot_product(self.weights, features)
}

///|
pub fn AdagradLogisticRegression::predict(
  self : AdagradLogisticRegression,
  features : Array[Double],
) -> Double {
  sigmoid(self.logit(features))
}

///|
pub fn AdagradLogisticRegression::predict_label(
  self : AdagradLogisticRegression,
  features : Array[Double],
  threshold? : Double = 0.5,
) -> Double {
  if self.predict(features) >= threshold {
    1.0
  } else {
    0.0
  }
}

///|
fn soft_threshold(value : Double, threshold : Double) -> Double {
  if value > threshold {
    value - threshold
  } else if value < -threshold {
    value + threshold
  } else {
    0.0
  }
}

///|
pub fn AdagradLogisticRegression::update(
  self : AdagradLogisticRegression,
  features : Array[Double],
  label : Double,
) -> Unit {
  self.update_weighted(features, label, 1.0)
}

///|
pub fn AdagradLogisticRegression::update_weighted(
  self : AdagradLogisticRegression,
  features : Array[Double],
  label : Double,
  sample_weight : Double,
) -> Unit {
  let prediction = self.predict(features)
  let error = (prediction - clamp(label, 0.0, 1.0)) * sample_weight
  let limit = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  self.steps += 1
  self.seen += sample_weight
  for i in 0..<limit {
    let gradient = error * features[i] + self.l2 * self.weights[i]
    self.accumulator[i] += gradient * gradient
    let adaptive = self.learning_rate *
      gradient /
      (self.accumulator[i].sqrt() + self.epsilon)
    let regularized = self.weights[i] - adaptive
    self.weights[i] = soft_threshold(
      regularized,
      self.learning_rate * self.l1 / (self.accumulator[i].sqrt() + self.epsilon),
    )
  }
}

///|
pub fn AdagradLogisticRegression::loss(
  self : AdagradLogisticRegression,
  features : Array[Double],
  label : Double,
) -> Double {
  let data_loss = binary_cross_entropy(self.predict(features), label)
  let regularization = self.l1 * l1_norm(self.weights) +
    0.5 * self.l2 * squared_norm(self.weights)
  data_loss + regularization
}

///|
pub fn AdagradLogisticRegression::regularization(
  self : AdagradLogisticRegression,
) -> Double {
  self.l1 * l1_norm(self.weights) + 0.5 * self.l2 * squared_norm(self.weights)
}

///|
pub fn AdagradLogisticRegression::feature_importance(
  self : AdagradLogisticRegression,
) -> Array[Double] {
  self.weights.map(value => if value < 0.0 { -value } else { value })
}

///|
pub fn AdagradLogisticRegression::sparsity(
  self : AdagradLogisticRegression,
  tolerance? : Double = 1.0e-12,
) -> Double {
  if self.weights.is_empty() {
    1.0
  } else {
    let zeros = self.weights.count_if(value => {
      let magnitude = if value < 0.0 { -value } else { value }
      magnitude <= tolerance
    })
    zeros.to_double() / self.weights.length().to_double()
  }
}

///|
pub fn AdagradLogisticRegression::reset(
  self : AdagradLogisticRegression,
) -> Unit {
  self.weights.fill(0.0)
  self.accumulator.fill(0.0)
  self.steps = 0
  self.seen = 0.0
}

///|
/// A linear regressor with Adagrad updates, useful when labels are continuous
/// and the stream has heterogeneous feature scales.
pub struct AdagradLinearRegression {
  weights : Array[Double]
  accumulator : Array[Double]
  learning_rate : Double
  epsilon : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn AdagradLinearRegression::new(
  dimension : Int,
  learning_rate? : Double = 0.05,
  epsilon? : Double = 1.0e-8,
  l2? : Double = 0.0,
) -> AdagradLinearRegression {
  let size = if dimension < 0 { 0 } else { dimension }
  {
    weights: Array::make(size, 0.0),
    accumulator: Array::make(size, 0.0),
    learning_rate,
    epsilon,
    l2,
    steps: 0,
  }
}

///|
pub fn AdagradLinearRegression::dimension(
  self : AdagradLinearRegression,
) -> Int {
  self.weights.length()
}

///|
pub fn AdagradLinearRegression::weights(
  self : AdagradLinearRegression,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn AdagradLinearRegression::predict(
  self : AdagradLinearRegression,
  features : Array[Double],
) -> Double {
  dot_product(self.weights, features)
}

///|
pub fn AdagradLinearRegression::update(
  self : AdagradLinearRegression,
  features : Array[Double],
  label : Double,
) -> Unit {
  let error = self.predict(features) - label
  let limit = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  for i in 0..<limit {
    let gradient = error * features[i] + self.l2 * self.weights[i]
    self.accumulator[i] += gradient * gradient
    self.weights[i] -= self.learning_rate *
      gradient /
      (self.accumulator[i].sqrt() + self.epsilon)
  }
  self.steps += 1
}

///|
pub fn AdagradLinearRegression::loss(
  self : AdagradLinearRegression,
  features : Array[Double],
  label : Double,
) -> Double {
  0.5 * squared_error(self.predict(features), label) +
  0.5 * self.l2 * squared_norm(self.weights)
}

///|
pub fn AdagradLinearRegression::residual(
  self : AdagradLinearRegression,
  features : Array[Double],
  label : Double,
) -> Double {
  label - self.predict(features)
}

///|
pub fn AdagradLinearRegression::steps(self : AdagradLinearRegression) -> Int {
  self.steps
}

///|
pub fn AdagradLinearRegression::reset(self : AdagradLinearRegression) -> Unit {
  self.weights.fill(0.0)
  self.accumulator.fill(0.0)
  self.steps = 0
}
