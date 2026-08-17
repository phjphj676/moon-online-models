///|
/// Online multiclass softmax regression.
///
/// The implementation uses a stable log-sum-exp softmax and updates all
/// classes for each event. This makes it suitable for streaming routing,
/// intent classification, and low-latency edge inference.
pub struct OnlineSoftmaxRegression {
  weights : Array[Array[Double]]
  learning_rate : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn OnlineSoftmaxRegression::new(
  classes : Int,
  dimension : Int,
  learning_rate? : Double = 0.05,
  l2? : Double = 0.0,
) -> OnlineSoftmaxRegression {
  let safe_classes = if classes < 0 { 0 } else { classes }
  let safe_dimension = if dimension < 0 { 0 } else { dimension }
  {
    weights: Array::makei(safe_classes, _ => Array::make(safe_dimension, 0.0)),
    learning_rate,
    l2,
    steps: 0,
  }
}

///|
pub fn OnlineSoftmaxRegression::classes(self : OnlineSoftmaxRegression) -> Int {
  self.weights.length()
}

///|
pub fn OnlineSoftmaxRegression::dimension(
  self : OnlineSoftmaxRegression,
) -> Int {
  if self.weights.is_empty() {
    0
  } else {
    self.weights[0].length()
  }
}

///|
pub fn OnlineSoftmaxRegression::steps(self : OnlineSoftmaxRegression) -> Int {
  self.steps
}

///|
pub fn OnlineSoftmaxRegression::weights(
  self : OnlineSoftmaxRegression,
) -> Array[Array[Double]] {
  self.weights.map(row => copy_vector(row))
}

///|
pub fn OnlineSoftmaxRegression::logits(
  self : OnlineSoftmaxRegression,
  features : Array[Double],
) -> Array[Double] {
  self.weights.map(row => dot_product(row, features))
}

///|
pub fn OnlineSoftmaxRegression::predict_proba(
  self : OnlineSoftmaxRegression,
  features : Array[Double],
) -> Array[Double] {
  probabilities_from_logits(self.logits(features))
}

///|
pub fn OnlineSoftmaxRegression::predict_class(
  self : OnlineSoftmaxRegression,
  features : Array[Double],
) -> Int? {
  argmax(self.predict_proba(features))
}

///|
pub fn OnlineSoftmaxRegression::predict_top_k(
  self : OnlineSoftmaxRegression,
  features : Array[Double],
  k : Int,
) -> Array[Int] {
  let probabilities = self.predict_proba(features)
  let order = Array::makei(probabilities.length(), i => i)
  order.sort_by((left, right) => {
    if probabilities[left] > probabilities[right] {
      -1
    } else if probabilities[left] < probabilities[right] {
      1
    } else {
      left - right
    }
  })
  let limit = if k < 0 {
    0
  } else if k > order.length() {
    order.length()
  } else {
    k
  }
  Array::makei(limit, i => order[i])
}

///|
pub fn OnlineSoftmaxRegression::update(
  self : OnlineSoftmaxRegression,
  features : Array[Double],
  label : Int,
) -> Bool {
  if label < 0 || label >= self.weights.length() {
    false
  } else {
    let probabilities = self.predict_proba(features)
    for class_index in 0..<self.weights.length() {
      let error = probabilities[class_index] -
        (if class_index == label { 1.0 } else { 0.0 })
      let row = self.weights[class_index]
      let limit = if row.length() < features.length() {
        row.length()
      } else {
        features.length()
      }
      for i in 0..<limit {
        row[i] -= self.learning_rate * (error * features[i] + self.l2 * row[i])
      }
    }
    self.steps += 1
    true
  }
}

///|
pub fn OnlineSoftmaxRegression::update_weighted(
  self : OnlineSoftmaxRegression,
  features : Array[Double],
  label : Int,
  weight : Double,
) -> Bool {
  if label < 0 || label >= self.weights.length() {
    false
  } else {
    let probabilities = self.predict_proba(features)
    for class_index in 0..<self.weights.length() {
      let error = weight *
        (
          probabilities[class_index] -
          (if class_index == label { 1.0 } else { 0.0 })
        )
      let row = self.weights[class_index]
      let limit = if row.length() < features.length() {
        row.length()
      } else {
        features.length()
      }
      for i in 0..<limit {
        row[i] -= self.learning_rate * (error * features[i] + self.l2 * row[i])
      }
    }
    self.steps += 1
    true
  }
}

///|
pub fn OnlineSoftmaxRegression::loss(
  self : OnlineSoftmaxRegression,
  features : Array[Double],
  label : Int,
) -> Double {
  if label < 0 || label >= self.weights.length() {
    0.0
  } else {
    let probabilities = self.predict_proba(features)
    -@math.ln(clamp_probability(probabilities[label])) +
    0.5 * self.l2 * self.weight_norm_squared()
  }
}

///|
pub fn OnlineSoftmaxRegression::weight_norm_squared(
  self : OnlineSoftmaxRegression,
) -> Double {
  let mut total = 0.0
  for row in self.weights {
    total += squared_norm(row)
  }
  total
}

///|
pub fn OnlineSoftmaxRegression::accuracy_on(
  self : OnlineSoftmaxRegression,
  samples : Array[Array[Double]],
  labels : Array[Int],
) -> Double {
  let size = if samples.length() < labels.length() {
    samples.length()
  } else {
    labels.length()
  }
  if size == 0 {
    0.0
  } else {
    let mut correct = 0
    for i in 0..<size {
      if self.predict_class(samples[i]) == Some(labels[i]) {
        correct += 1
      }
    }
    correct.to_double() / size.to_double()
  }
}

///|
pub fn OnlineSoftmaxRegression::top_k_accuracy(
  self : OnlineSoftmaxRegression,
  samples : Array[Array[Double]],
  labels : Array[Int],
  k : Int,
) -> Double {
  let size = if samples.length() < labels.length() {
    samples.length()
  } else {
    labels.length()
  }
  if size == 0 {
    0.0
  } else {
    let mut correct = 0
    for i in 0..<size {
      if self.predict_top_k(samples[i], k).contains(labels[i]) {
        correct += 1
      }
    }
    correct.to_double() / size.to_double()
  }
}

///|
pub fn OnlineSoftmaxRegression::reset(self : OnlineSoftmaxRegression) -> Unit {
  for row in self.weights {
    row.fill(0.0)
  }
  self.steps = 0
}

///|
pub struct ClassCountTracker {
  counts : Array[Int]
}

///|
pub fn ClassCountTracker::new(classes : Int) -> ClassCountTracker {
  { counts: Array::make(if classes < 0 { 0 } else { classes }, 0) }
}

///|
pub fn ClassCountTracker::observe(
  self : ClassCountTracker,
  label : Int,
) -> Bool {
  if label < 0 || label >= self.counts.length() {
    false
  } else {
    self.counts[label] += 1
    true
  }
}

///|
pub fn ClassCountTracker::count(self : ClassCountTracker, label : Int) -> Int {
  self.counts.get(label).unwrap_or(0)
}

///|
pub fn ClassCountTracker::total(self : ClassCountTracker) -> Int {
  self.counts.fold(init=0, (total, value) => total + value)
}

///|
pub fn ClassCountTracker::prior(
  self : ClassCountTracker,
  label : Int,
  smoothing? : Double = 1.0,
) -> Double {
  let denominator = self.total().to_double() +
    smoothing * self.counts.length().to_double()
  if denominator <= 0.0 {
    0.0
  } else {
    (self.count(label).to_double() + smoothing) / denominator
  }
}

///|
pub fn ClassCountTracker::priors(
  self : ClassCountTracker,
  smoothing? : Double = 1.0,
) -> Array[Double] {
  Array::makei(self.counts.length(), i => self.prior(i, smoothing~))
}

///|
pub fn ClassCountTracker::counts(self : ClassCountTracker) -> Array[Int] {
  self.counts.map(value => value)
}
