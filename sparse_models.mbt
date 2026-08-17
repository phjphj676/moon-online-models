///|
/// FTRL-Proximal for sparse vectors. Only non-zero coordinates are touched
/// during an update, which keeps the per-event cost O(nnz).
pub struct SparseFTRL {
  dimension : Int
  alpha : Double
  beta : Double
  l1 : Double
  l2 : Double
  z : Array[Double]
  n : Array[Double]
  mut steps : Int
}

///|
pub fn SparseFTRL::new(
  dimension : Int,
  alpha? : Double = 0.1,
  beta? : Double = 1.0,
  l1? : Double = 1.0,
  l2? : Double = 1.0,
) -> SparseFTRL {
  let size = if dimension < 0 { 0 } else { dimension }
  {
    dimension: size,
    alpha,
    beta,
    l1,
    l2,
    z: Array::make(size, 0.0),
    n: Array::make(size, 0.0),
    steps: 0,
  }
}

///|
fn SparseFTRL::sparse_ftrl_weight(self : SparseFTRL, index : Int) -> Double {
  let value = self.z[index]
  let sign = if value < 0.0 { -1.0 } else { 1.0 }
  if sign * value <= self.l1 {
    0.0
  } else {
    (sign * self.l1 - value) /
    ((self.beta + self.n[index].sqrt()) / self.alpha + self.l2)
  }
}

///|
pub fn SparseFTRL::predict(
  self : SparseFTRL,
  features : SparseVector,
) -> Double {
  sigmoid(
    features
    .entries()
    .fold(init=0.0, (total, entry) => total + self.sparse_weighted_entry(entry)),
  )
}

///|
fn SparseFTRL::sparse_weighted_entry(
  self : SparseFTRL,
  entry : SparseEntry,
) -> Double {
  if entry.index() < 0 || entry.index() >= self.dimension {
    0.0
  } else {
    self.sparse_ftrl_weight(entry.index()) * entry.value()
  }
}

///|
pub fn SparseFTRL::update(
  self : SparseFTRL,
  features : SparseVector,
  label : Double,
) -> Unit {
  let prediction = self.predict(features)
  let error = prediction - clamp(label, 0.0, 1.0)
  for entry in features.entries() {
    let index = entry.index()
    if index >= 0 && index < self.dimension {
      let gradient = error * entry.value()
      let sigma = (self.n[index] + gradient * gradient).sqrt() -
        self.n[index].sqrt()
      self.z[index] += gradient - sigma * self.sparse_ftrl_weight(index)
      self.n[index] += gradient * gradient
    }
  }
  self.steps += 1
}

///|
pub fn SparseFTRL::dimension(self : SparseFTRL) -> Int {
  self.dimension
}

///|
pub fn SparseFTRL::steps(self : SparseFTRL) -> Int {
  self.steps
}

///|
pub fn SparseFTRL::weight(self : SparseFTRL, index : Int) -> Double {
  if index < 0 || index >= self.dimension {
    0.0
  } else {
    self.sparse_ftrl_weight(index)
  }
}

///|
pub fn SparseFTRL::non_zero_weights(
  self : SparseFTRL,
  tolerance? : Double = 1.0e-12,
) -> SparseVector {
  let entries = Array::make(0, SparseEntry::new(0, 0.0))
  for i in 0..<self.dimension {
    let value = self.sparse_ftrl_weight(i)
    if value.abs() > tolerance {
      entries.push(SparseEntry::new(i, value))
    }
  }
  SparseVector::from_entries(self.dimension, entries)
}

///|
pub fn SparseFTRL::reset(self : SparseFTRL) -> Unit {
  self.z.fill(0.0)
  self.n.fill(0.0)
  self.steps = 0
}

///|
/// Sparse Adagrad logistic regression with a dense accumulator for predictable
/// inference and sparse updates.
pub struct SparseAdagradClassifier {
  weights : Array[Double]
  accumulator : Array[Double]
  learning_rate : Double
  epsilon : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn SparseAdagradClassifier::new(
  dimension : Int,
  learning_rate? : Double = 0.05,
  epsilon? : Double = 1.0e-8,
  l2? : Double = 0.0,
) -> SparseAdagradClassifier {
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
pub fn SparseAdagradClassifier::predict(
  self : SparseAdagradClassifier,
  features : SparseVector,
) -> Double {
  sigmoid(features.dot_dense(self.weights))
}

///|
pub fn SparseAdagradClassifier::update(
  self : SparseAdagradClassifier,
  features : SparseVector,
  label : Double,
) -> Unit {
  let error = self.predict(features) - clamp(label, 0.0, 1.0)
  for entry in features.entries() {
    let index = entry.index()
    if index >= 0 && index < self.weights.length() {
      let gradient = error * entry.value() + self.l2 * self.weights[index]
      self.accumulator[index] += gradient * gradient
      self.weights[index] -= self.learning_rate *
        gradient /
        (self.accumulator[index].sqrt() + self.epsilon)
    }
  }
  self.steps += 1
}

///|
pub fn SparseAdagradClassifier::weight(
  self : SparseAdagradClassifier,
  index : Int,
) -> Double {
  self.weights.get(index).unwrap_or(0.0)
}

///|
pub fn SparseAdagradClassifier::weights(
  self : SparseAdagradClassifier,
) -> SparseVector {
  SparseVector::from_dense(self.weights, threshold=1.0e-12)
}

///|
pub fn SparseAdagradClassifier::steps(self : SparseAdagradClassifier) -> Int {
  self.steps
}

///|
pub fn SparseAdagradClassifier::reset(self : SparseAdagradClassifier) -> Unit {
  self.weights.fill(0.0)
  self.accumulator.fill(0.0)
  self.steps = 0
}
