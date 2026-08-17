///|
pub enum ObjectiveKind {
  Squared
  Logistic
  Huber(delta~ : Double)
  Quantile(probability~ : Double)
  Hinge
} derive(ToJson, FromJson, Debug, Eq)

///|
pub fn ObjectiveKind::loss(
  self : ObjectiveKind,
  prediction : Double,
  label : Double,
) -> Double {
  match self {
    Squared => 0.5 * squared_error(prediction, label)
    Logistic => binary_cross_entropy(sigmoid(prediction), label)
    Huber(delta~) => smooth_l1_loss(prediction - label, beta=delta)
    Quantile(probability~) => {
      let error = label - prediction
      if error >= 0.0 {
        probability * error
      } else {
        (probability - 1.0) * error
      }
    }
    Hinge => hinge_loss(prediction, if label >= 0.5 { 1.0 } else { -1.0 })
  }
}

///|
pub fn ObjectiveKind::gradient(
  self : ObjectiveKind,
  prediction : Double,
  label : Double,
) -> Double {
  match self {
    Squared => prediction - label
    Logistic => sigmoid(prediction) - clamp(label, 0.0, 1.0)
    Huber(delta~) => {
      let error = prediction - label
      clamp(error, -delta, delta)
    }
    Quantile(probability~) =>
      if label - prediction >= 0.0 {
        -probability
      } else {
        1.0 - probability
      }
    Hinge =>
      if 1.0 - prediction * (if label >= 0.5 { 1.0 } else { -1.0 }) > 0.0 {
        -(if label >= 0.5 { 1.0 } else { -1.0 })
      } else {
        0.0
      }
  }
}

///|
pub struct Regularizer {
  l1 : Double
  l2 : Double
}

///|
pub fn Regularizer::new(l1? : Double = 0.0, l2? : Double = 0.0) -> Regularizer {
  { l1: if l1 < 0.0 { 0.0 } else { l1 }, l2: if l2 < 0.0 { 0.0 } else { l2 } }
}

///|
pub fn Regularizer::penalty(
  self : Regularizer,
  weights : Array[Double],
) -> Double {
  self.l1 * l1_norm(weights) + 0.5 * self.l2 * squared_norm(weights)
}

///|
pub fn Regularizer::gradient(
  self : Regularizer,
  weights : Array[Double],
  index : Int,
) -> Double {
  let value = weights.get(index).unwrap_or(0.0)
  let sign = if value < 0.0 { -1.0 } else if value > 0.0 { 1.0 } else { 0.0 }
  self.l1 * sign + self.l2 * value
}

///|
pub fn Regularizer::proximal(
  self : Regularizer,
  value : Double,
  step : Double,
) -> Double {
  soft_threshold(value, self.l1 * step) / (1.0 + self.l2 * step)
}

///|
pub struct ObjectiveTracker {
  objective : ObjectiveKind
  mut count : Double
  mut total : Double
  mut last : Double
}

///|
pub fn ObjectiveTracker::new(objective : ObjectiveKind) -> ObjectiveTracker {
  { objective, count: 0.0, total: 0.0, last: 0.0 }
}

///|
pub fn ObjectiveTracker::observe(
  self : ObjectiveTracker,
  prediction : Double,
  label : Double,
  weight? : Double = 1.0,
) -> Unit {
  let loss = self.objective.loss(prediction, label)
  self.count += weight
  self.total += weight * loss
  self.last = loss
}

///|
pub fn ObjectiveTracker::mean(self : ObjectiveTracker) -> Double {
  if self.count <= 0.0 {
    0.0
  } else {
    self.total / self.count
  }
}

///|
pub fn ObjectiveTracker::last(self : ObjectiveTracker) -> Double {
  self.last
}

///|
pub fn ObjectiveTracker::count(self : ObjectiveTracker) -> Double {
  self.count
}

///|
pub fn ObjectiveTracker::reset(self : ObjectiveTracker) -> Unit {
  self.count = 0.0
  self.total = 0.0
  self.last = 0.0
}
