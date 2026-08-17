///|
/// Weighted probability ensemble with online expert reweighting.
pub struct WeightedProbabilityEnsemble {
  weights : Array[Double]
  mut observations : Int
}

///|
pub fn WeightedProbabilityEnsemble::new(
  experts : Int,
) -> WeightedProbabilityEnsemble {
  let size = if experts < 0 { 0 } else { experts }
  { weights: Array::make(size, 1.0), observations: 0 }
}

///|
pub fn WeightedProbabilityEnsemble::experts(
  self : WeightedProbabilityEnsemble,
) -> Int {
  self.weights.length()
}

///|
pub fn WeightedProbabilityEnsemble::predict(
  self : WeightedProbabilityEnsemble,
  predictions : Array[Double],
) -> Double {
  let size = if predictions.length() < self.weights.length() {
    predictions.length()
  } else {
    self.weights.length()
  }
  let mut numerator = 0.0
  let mut denominator = 0.0
  for i in 0..<size {
    numerator += self.weights[i] * predictions[i]
    denominator += self.weights[i]
  }
  if denominator <= 0.0 {
    0.5
  } else {
    clamp(numerator / denominator, 0.0, 1.0)
  }
}

///|
pub fn WeightedProbabilityEnsemble::update(
  self : WeightedProbabilityEnsemble,
  predictions : Array[Double],
  label : Double,
  learning_rate? : Double = 0.05,
) -> Unit {
  let size = if predictions.length() < self.weights.length() {
    predictions.length()
  } else {
    self.weights.length()
  }
  for i in 0..<size {
    let loss = binary_cross_entropy(predictions[i], label)
    self.weights[i] *= @math.exp(-learning_rate * loss)
    if self.weights[i] < 1.0e-12 {
      self.weights[i] = 1.0e-12
    }
  }
  self.observations += 1
}

///|
pub fn WeightedProbabilityEnsemble::weights(
  self : WeightedProbabilityEnsemble,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn WeightedProbabilityEnsemble::normalized_weights(
  self : WeightedProbabilityEnsemble,
) -> Array[Double] {
  let total = sum_values(self.weights)
  if total <= 0.0 {
    Array::make(self.weights.length(), 0.0)
  } else {
    scale_values(self.weights, 1.0 / total)
  }
}

///|
pub fn WeightedProbabilityEnsemble::observations(
  self : WeightedProbabilityEnsemble,
) -> Int {
  self.observations
}

///|
pub fn WeightedProbabilityEnsemble::reset(
  self : WeightedProbabilityEnsemble,
) -> Unit {
  self.weights.fill(1.0)
  self.observations = 0
}

///|
/// Online bagging around Adagrad logistic learners.
pub struct OnlineBaggingClassifier {
  models : Array[AdagradLogisticRegression]
  rng : DeterministicRng
  mut observations : Int
}

///|
pub fn OnlineBaggingClassifier::new(
  model_count : Int,
  dimension : Int,
  seed? : UInt64 = 1,
) -> OnlineBaggingClassifier {
  let count = if model_count < 0 { 0 } else { model_count }
  {
    models: Array::makei(count, _ => AdagradLogisticRegression::new(dimension)),
    rng: DeterministicRng::new(seed),
    observations: 0,
  }
}

///|
pub fn OnlineBaggingClassifier::models(self : OnlineBaggingClassifier) -> Int {
  self.models.length()
}

///|
pub fn OnlineBaggingClassifier::predict(
  self : OnlineBaggingClassifier,
  features : Array[Double],
) -> Double {
  if self.models.is_empty() {
    0.5
  } else {
    let mut total = 0.0
    for model in self.models {
      total += model.predict(features)
    }
    total / self.models.length().to_double()
  }
}

///|
pub fn OnlineBaggingClassifier::update(
  self : OnlineBaggingClassifier,
  features : Array[Double],
  label : Double,
) -> Unit {
  for model in self.models {
    let copies = poisson_one(self.rng)
    for _ in 0..<copies {
      model.update(features, label)
    }
  }
  self.observations += 1
}

///|
fn poisson_one(rng : DeterministicRng) -> Int {
  let mut count = 0
  if rng.next_double() < 0.37 {
    count += 1
  }
  if rng.next_double() < 0.18 {
    count += 1
  }
  if rng.next_double() < 0.08 {
    count += 1
  }
  count
}

///|
pub fn OnlineBaggingClassifier::observations(
  self : OnlineBaggingClassifier,
) -> Int {
  self.observations
}

///|
pub fn OnlineBaggingClassifier::member_predictions(
  self : OnlineBaggingClassifier,
  features : Array[Double],
) -> Array[Double] {
  self.models.map(model => model.predict(features))
}

///|
pub fn OnlineBaggingClassifier::reset(self : OnlineBaggingClassifier) -> Unit {
  for model in self.models {
    model.reset()
  }
  self.observations = 0
}
