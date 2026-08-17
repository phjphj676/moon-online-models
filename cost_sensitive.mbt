///|
/// Cost-sensitive decision rules for imbalanced streaming classification.
pub struct ClassCost {
  false_positive : Double
  false_negative : Double
  true_positive : Double
  true_negative : Double
}

///|
pub fn ClassCost::new(
  false_positive? : Double = 1.0,
  false_negative? : Double = 1.0,
  true_positive? : Double = 0.0,
  true_negative? : Double = 0.0,
) -> ClassCost {
  {
    false_positive: if false_positive < 0.0 {
      0.0
    } else {
      false_positive
    },
    false_negative: if false_negative < 0.0 {
      0.0
    } else {
      false_negative
    },
    true_positive,
    true_negative,
  }
}

///|
pub fn ClassCost::cost(
  self : ClassCost,
  prediction : Double,
  label : Double,
  threshold : Double,
) -> Double {
  let predicted_positive = prediction >= threshold
  let actual_positive = label >= 0.5
  if predicted_positive && actual_positive {
    self.true_positive
  } else if predicted_positive {
    self.false_positive
  } else if actual_positive {
    self.false_negative
  } else {
    self.true_negative
  }
}

///|
pub fn ClassCost::expected_cost(
  self : ClassCost,
  probability : Double,
  threshold : Double,
) -> Double {
  probability * self.cost(1.0, 1.0, threshold) +
  (1.0 - probability) * self.cost(1.0, 0.0, threshold)
}

///|
pub fn ClassCost::positive_weight(self : ClassCost) -> Double {
  self.false_negative + self.true_positive.abs()
}

///|
pub fn ClassCost::negative_weight(self : ClassCost) -> Double {
  self.false_positive + self.true_negative.abs()
}

///|
pub fn ClassCost::balanced_threshold(self : ClassCost) -> Double {
  let positive = self.positive_weight()
  let negative = self.negative_weight()
  if positive + negative <= 0.0 {
    0.5
  } else {
    clamp(negative / (positive + negative), 0.0, 1.0)
  }
}

///|
pub fn cost_sensitive_decision(probability : Double, costs : ClassCost) -> Bool {
  probability >= costs.balanced_threshold()
}

///|
pub fn cost_matrix(costs : ClassCost) -> Array[Array[Double]] {
  [
    [costs.true_negative, costs.false_positive],
    [costs.false_negative, costs.true_positive],
  ]
}

///|
pub fn expected_binary_cost(
  probability : Double,
  positive_cost : Double,
  negative_cost : Double,
) -> Double {
  let p = clamp_probability(probability)
  p * positive_cost + (1.0 - p) * negative_cost
}

///|
pub fn threshold_is_valid(threshold : Double) -> Bool {
  threshold >= 0.0 && threshold <= 1.0
}

///|
pub fn safe_threshold(threshold : Double) -> Double {
  clamp(threshold, 0.0, 1.0)
}

///|
pub struct CostSensitiveEvaluator {
  costs : ClassCost
  mut total : Double
  mut count : Int
}

///|
pub fn CostSensitiveEvaluator::new(
  costs? : ClassCost = ClassCost::new(),
) -> CostSensitiveEvaluator {
  { costs, total: 0.0, count: 0 }
}

///|
pub fn CostSensitiveEvaluator::observe(
  self : CostSensitiveEvaluator,
  prediction : Double,
  label : Double,
  threshold : Double,
) -> Double {
  let value = self.costs.cost(prediction, label, safe_threshold(threshold))
  self.total += value
  self.count += 1
  value
}

///|
pub fn CostSensitiveEvaluator::mean(self : CostSensitiveEvaluator) -> Double {
  if self.count == 0 {
    0.0
  } else {
    self.total / self.count.to_double()
  }
}

///|
pub fn CostSensitiveEvaluator::count(self : CostSensitiveEvaluator) -> Int {
  self.count
}

///|
pub fn CostSensitiveEvaluator::total(self : CostSensitiveEvaluator) -> Double {
  self.total
}

///|
pub fn CostSensitiveEvaluator::reset(self : CostSensitiveEvaluator) -> Unit {
  self.total = 0.0
  self.count = 0
}

///|
pub struct ThresholdOptimizer {
  costs : ClassCost
  minimum : Double
  maximum : Double
  steps : Int
  mut best_threshold : Double
  mut best_cost : Double
}

///|
pub fn ThresholdOptimizer::new(
  costs? : ClassCost = ClassCost::new(),
  minimum? : Double = 0.01,
  maximum? : Double = 0.99,
  steps? : Int = 99,
) -> ThresholdOptimizer {
  {
    costs,
    minimum: clamp(minimum, 0.0, 1.0),
    maximum: clamp(maximum, 0.0, 1.0),
    steps: if steps < 1 {
      1
    } else {
      steps
    },
    best_threshold: 0.5,
    best_cost: 0.0,
  }
}

///|
pub fn ThresholdOptimizer::fit(
  self : ThresholdOptimizer,
  predictions : Array[Double],
  labels : Array[Double],
) -> Double {
  let count = if predictions.length() < labels.length() {
    predictions.length()
  } else {
    labels.length()
  }
  let mut best = 0.5
  let mut cost = 1.0e30
  for step in 0..<=self.steps {
    let threshold = self.minimum +
      (self.maximum - self.minimum) * step.to_double() / self.steps.to_double()
    let mut candidate = 0.0
    for i in 0..<count {
      candidate += self.costs.cost(predictions[i], labels[i], threshold)
    }
    if candidate < cost {
      cost = candidate
      best = threshold
    }
  }
  self.best_threshold = best
  self.best_cost = if count == 0 { 0.0 } else { cost / count.to_double() }
  best
}

///|
pub fn ThresholdOptimizer::threshold(self : ThresholdOptimizer) -> Double {
  self.best_threshold
}

///|
pub fn ThresholdOptimizer::cost(self : ThresholdOptimizer) -> Double {
  self.best_cost
}

///|
pub fn ThresholdOptimizer::reset(self : ThresholdOptimizer) -> Unit {
  self.best_threshold = 0.5
  self.best_cost = 0.0
}
