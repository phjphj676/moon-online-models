///|
/// Shared numerical helpers used by the online learning models.
///
/// The helpers in this file intentionally avoid hidden allocations in hot
/// paths. They also provide safe variants for callers that receive data from
/// an external stream and cannot assume that dimensions are correct.

///|
pub struct ValidationReport {
  valid : Bool
  message : String
} derive(ToJson, FromJson, Debug, Eq)

///|
pub fn ValidationReport::ok() -> ValidationReport {
  { valid: true, message: "ok" }
}

///|
pub fn ValidationReport::error(message : String) -> ValidationReport {
  { valid: false, message }
}

///|
pub fn ValidationReport::is_valid(self : ValidationReport) -> Bool {
  self.valid
}

///|
pub fn ValidationReport::message(self : ValidationReport) -> String {
  self.message
}

///|
pub fn valid_dimension(dim : Int) -> Bool {
  dim >= 0
}

///|
pub fn validate_dimension(dim : Int) -> ValidationReport {
  if dim < 0 {
    ValidationReport::error("dimension must be non-negative")
  } else {
    ValidationReport::ok()
  }
}

///|
pub fn validate_vector(features : Array[Double], dim : Int) -> ValidationReport {
  if dim < 0 {
    ValidationReport::error("dimension must be non-negative")
  } else if features.length() != dim {
    ValidationReport::error(
      "feature dimension mismatch: expected {dim}, got {features.length()}",
    )
  } else {
    ValidationReport::ok()
  }
}

///|
pub fn validate_probability(value : Double) -> ValidationReport {
  if value < 0.0 || value > 1.0 {
    ValidationReport::error("probability must be in [0, 1]")
  } else {
    ValidationReport::ok()
  }
}

///|
pub fn validate_label(value : Double) -> ValidationReport {
  if value == 0.0 || value == 1.0 {
    ValidationReport::ok()
  } else {
    ValidationReport::error("binary label must be 0 or 1")
  }
}

///|
pub fn clamp(value : Double, lower : Double, upper : Double) -> Double {
  if value < lower {
    lower
  } else if value > upper {
    upper
  } else {
    value
  }
}

///|
pub fn clamp_probability(value : Double) -> Double {
  clamp(value, 1.0e-15, 1.0 - 1.0e-15)
}

///|
pub fn sigmoid(value : Double) -> Double {
  if value >= 0.0 {
    1.0 / (1.0 + @math.exp(-value))
  } else {
    let e = @math.exp(value)
    e / (1.0 + e)
  }
}

///|
pub fn softplus(value : Double) -> Double {
  if value > 20.0 {
    value
  } else if value < -20.0 {
    @math.exp(value)
  } else {
    @math.ln(1.0 + @math.exp(value))
  }
}

///|
pub fn logit(probability : Double) -> Double {
  let p = clamp_probability(probability)
  @math.ln(p / (1.0 - p))
}

///|
pub fn copy_vector(values : Array[Double]) -> Array[Double] {
  Array::makei(values.length(), i => values[i])
}

///|
pub fn dot_product(left : Array[Double], right : Array[Double]) -> Double {
  let limit = if left.length() < right.length() {
    left.length()
  } else {
    right.length()
  }
  let mut total = 0.0
  for i in 0..<limit {
    total += left[i] * right[i]
  }
  total
}

///|
pub fn dot_product_checked(
  left : Array[Double],
  right : Array[Double],
) -> Double? {
  if left.length() != right.length() {
    None
  } else {
    Some(dot_product(left, right))
  }
}

///|
pub fn squared_norm(values : Array[Double]) -> Double {
  dot_product(values, values)
}

///|
pub fn l1_norm(values : Array[Double]) -> Double {
  let mut total = 0.0
  for value in values {
    total += if value < 0.0 { -value } else { value }
  }
  total
}

///|
pub fn max_abs(values : Array[Double]) -> Double {
  let mut result = 0.0
  for value in values {
    let magnitude = if value < 0.0 { -value } else { value }
    if magnitude > result {
      result = magnitude
    }
  }
  result
}

///|
pub fn sum_values(values : Array[Double]) -> Double {
  let mut total = 0.0
  for value in values {
    total += value
  }
  total
}

///|
pub fn mean_values(values : Array[Double]) -> Double {
  if values.is_empty() {
    0.0
  } else {
    sum_values(values) / values.length().to_double()
  }
}

///|
pub fn variance_values(values : Array[Double]) -> Double {
  if values.length() < 2 {
    0.0
  } else {
    let mean = mean_values(values)
    let mut total = 0.0
    for value in values {
      let delta = value - mean
      total += delta * delta
    }
    total / (values.length() - 1).to_double()
  }
}

///|
pub fn standard_deviation(values : Array[Double]) -> Double {
  let variance = variance_values(values)
  if variance <= 0.0 {
    0.0
  } else {
    variance.sqrt()
  }
}

///|
pub fn add_scaled_in_place(
  target : Array[Double],
  source : Array[Double],
  scale : Double,
) -> Unit {
  let limit = if target.length() < source.length() {
    target.length()
  } else {
    source.length()
  }
  for i in 0..<limit {
    target[i] += scale * source[i]
  }
}

///|
pub fn subtract_scaled_in_place(
  target : Array[Double],
  source : Array[Double],
  scale : Double,
) -> Unit {
  let limit = if target.length() < source.length() {
    target.length()
  } else {
    source.length()
  }
  for i in 0..<limit {
    target[i] -= scale * source[i]
  }
}

///|
pub fn scale_values(values : Array[Double], scale : Double) -> Array[Double] {
  Array::makei(values.length(), i => values[i] * scale)
}

///|
pub fn add_values(left : Array[Double], right : Array[Double]) -> Array[Double] {
  let size = if left.length() > right.length() {
    left.length()
  } else {
    right.length()
  }
  Array::makei(size, i => {
    let a = left.get(i).unwrap_or(0.0)
    let b = right.get(i).unwrap_or(0.0)
    a + b
  })
}

///|
pub fn subtract_values(
  left : Array[Double],
  right : Array[Double],
) -> Array[Double] {
  let size = if left.length() > right.length() {
    left.length()
  } else {
    right.length()
  }
  Array::makei(size, i => {
    let a = left.get(i).unwrap_or(0.0)
    let b = right.get(i).unwrap_or(0.0)
    a - b
  })
}

///|
pub fn hadamard_product(
  left : Array[Double],
  right : Array[Double],
) -> Array[Double] {
  let size = if left.length() < right.length() {
    left.length()
  } else {
    right.length()
  }
  Array::makei(size, i => left[i] * right[i])
}

///|
pub fn normalize_l2(values : Array[Double]) -> Array[Double] {
  let length = squared_norm(values).sqrt()
  if length <= 1.0e-15 {
    copy_vector(values)
  } else {
    scale_values(values, 1.0 / length)
  }
}

///|
pub fn cosine_similarity(left : Array[Double], right : Array[Double]) -> Double {
  let denominator = squared_norm(left).sqrt() * squared_norm(right).sqrt()
  if denominator <= 1.0e-15 {
    0.0
  } else {
    dot_product(left, right) / denominator
  }
}

///|
pub fn argmax(values : Array[Double]) -> Int? {
  if values.is_empty() {
    None
  } else {
    let mut best_index = 0
    let mut best_value = values[0]
    for i in 1..<values.length() {
      if values[i] > best_value {
        best_index = i
        best_value = values[i]
      }
    }
    Some(best_index)
  }
}

///|
pub fn top_index(values : Array[Double], rank : Int) -> Int? {
  if rank < 0 || rank >= values.length() {
    None
  } else {
    let order = Array::makei(values.length(), i => i)
    order.sort_by((left, right) => {
      if values[left] > values[right] {
        -1
      } else if values[left] < values[right] {
        1
      } else {
        left - right
      }
    })
    Some(order[rank])
  }
}

///|
pub fn one_hot(index : Int, size : Int) -> Array[Double] {
  let result = Array::make(if size < 0 { 0 } else { size }, 0.0)
  if index >= 0 && index < result.length() {
    result[index] = 1.0
  }
  result
}

///|
pub fn probabilities_from_logits(logits : Array[Double]) -> Array[Double] {
  if logits.is_empty() {
    []
  } else {
    let mut maximum = logits[0]
    for value in logits {
      if value > maximum {
        maximum = value
      }
    }
    let values = Array::makei(logits.length(), i => {
      @math.exp(logits[i] - maximum)
    })
    let denominator = sum_values(values)
    if denominator <= 0.0 {
      Array::make(logits.length(), 1.0 / logits.length().to_double())
    } else {
      scale_values(values, 1.0 / denominator)
    }
  }
}

///|
pub fn binary_cross_entropy(probability : Double, label : Double) -> Double {
  let p = clamp_probability(probability)
  -(label * @math.ln(p) + (1.0 - label) * @math.ln(1.0 - p))
}

///|
pub fn squared_error(prediction : Double, label : Double) -> Double {
  let error = prediction - label
  error * error
}

///|
pub fn absolute_error(prediction : Double, label : Double) -> Double {
  let error = prediction - label
  if error < 0.0 {
    -error
  } else {
    error
  }
}

///|
pub fn hinge_loss(score : Double, label : Double) -> Double {
  let margin = 1.0 - score * label
  if margin > 0.0 {
    margin
  } else {
    0.0
  }
}

///|
pub fn smooth_l1_loss(error : Double, beta? : Double = 1.0) -> Double {
  let magnitude = if error < 0.0 { -error } else { error }
  if magnitude < beta {
    0.5 * error * error / beta
  } else {
    magnitude - 0.5 * beta
  }
}

///|
pub fn r2_score(predictions : Array[Double], labels : Array[Double]) -> Double {
  if predictions.is_empty() || predictions.length() != labels.length() {
    0.0
  } else {
    let mean = mean_values(labels)
    let mut residual = 0.0
    let mut total = 0.0
    for i in 0..<labels.length() {
      residual += squared_error(predictions[i], labels[i])
      total += squared_error(labels[i], mean)
    }
    if total <= 1.0e-15 {
      0.0
    } else {
      1.0 - residual / total
    }
  }
}

///|
pub fn all_close(
  left : Array[Double],
  right : Array[Double],
  tolerance? : Double = 1.0e-9,
) -> Bool {
  if left.length() != right.length() {
    false
  } else {
    let mut result = true
    for i in 0..<left.length() {
      if (left[i] - right[i]).abs() > tolerance {
        result = false
      }
    }
    result
  }
}

///|
pub fn is_non_decreasing(values : Array[Double]) -> Bool {
  let mut result = true
  for i in 1..<values.length() {
    if values[i] < values[i - 1] {
      result = false
    }
  }
  result
}

///|
pub fn cumulative_sum(values : Array[Double]) -> Array[Double] {
  let result = Array::make(values.length(), 0.0)
  let mut total = 0.0
  for i in 0..<values.length() {
    total += values[i]
    result[i] = total
  }
  result
}

///|
pub fn weighted_mean(values : Array[Double], weights : Array[Double]) -> Double {
  let size = if values.length() < weights.length() {
    values.length()
  } else {
    weights.length()
  }
  let mut weighted = 0.0
  let mut total_weight = 0.0
  for i in 0..<size {
    weighted += values[i] * weights[i]
    total_weight += weights[i]
  }
  if total_weight <= 0.0 {
    0.0
  } else {
    weighted / total_weight
  }
}

///|
pub fn weighted_variance(
  values : Array[Double],
  weights : Array[Double],
) -> Double {
  let mean = weighted_mean(values, weights)
  let size = if values.length() < weights.length() {
    values.length()
  } else {
    weights.length()
  }
  let mut weighted = 0.0
  let mut total_weight = 0.0
  for i in 0..<size {
    let error = values[i] - mean
    weighted += weights[i] * error * error
    total_weight += weights[i]
  }
  if total_weight <= 0.0 {
    0.0
  } else {
    weighted / total_weight
  }
}

///|
pub fn moving_average(
  previous : Double,
  value : Double,
  smoothing : Double,
) -> Double {
  let rate = clamp(smoothing, 0.0, 1.0)
  rate * value + (1.0 - rate) * previous
}
