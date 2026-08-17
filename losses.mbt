///|
/// Loss functions and defensive gradient utilities shared by online learners.
/// The implementations are allocation-light and keep all numerical guards at
/// the package boundary so a malformed event cannot poison a long-running job.
pub(all) enum LossKind {
  Squared
  Absolute
  Huber
  LogLoss
  Hinge
  Quantile
  Poisson
} derive(Debug, Eq)

///|
pub fn loss_kind_catalog() -> Array[LossKind] {
  [Squared, Absolute, Huber, LogLoss, Hinge, Quantile, Poisson]
}

///|
pub fn loss_value(
  kind : LossKind,
  prediction : Double,
  label : Double,
  parameter? : Double = 1.0,
) -> Double {
  match kind {
    Squared => {
      let error = prediction - label
      0.5 * error * error
    }
    Absolute => (prediction - label).abs()
    Huber => {
      let delta = if parameter <= 0.0 { 1.0 } else { parameter }
      let error = (prediction - label).abs()
      if error <= delta {
        0.5 * error * error
      } else {
        delta * (error - 0.5 * delta)
      }
    }
    LogLoss => {
      let probability = clamp_probability(prediction)
      -(label * @math.ln(probability) +
      (1.0 - label) * @math.ln(1.0 - probability))
    }
    Hinge => {
      let margin = label * prediction
      if margin >= 1.0 {
        0.0
      } else {
        1.0 - margin
      }
    }
    Quantile => {
      let quantile = clamp(parameter, 0.0, 1.0)
      let error = label - prediction
      if error >= 0.0 {
        quantile * error
      } else {
        (quantile - 1.0) * error
      }
    }
    Poisson => {
      let rate = if prediction < 1.0e-12 { 1.0e-12 } else { prediction }
      rate - label * @math.ln(rate)
    }
  }
}

///|
pub fn loss_gradient(
  kind : LossKind,
  prediction : Double,
  label : Double,
  parameter? : Double = 1.0,
) -> Double {
  match kind {
    Squared => prediction - label
    Absolute => if prediction >= label { 1.0 } else { -1.0 }
    Huber => {
      let delta = if parameter <= 0.0 { 1.0 } else { parameter }
      let error = prediction - label
      if error > delta {
        delta
      } else if error < -delta {
        -delta
      } else {
        error
      }
    }
    LogLoss => clamp_probability(prediction) - label
    Hinge => if label * prediction < 1.0 { -label } else { 0.0 }
    Quantile => {
      let quantile = clamp(parameter, 0.0, 1.0)
      if label > prediction {
        -quantile
      } else {
        1.0 - quantile
      }
    }
    Poisson => @math.exp(prediction) - label
  }
}

///|
pub fn logits_to_probability(logit : Double) -> Double {
  sigmoid(logit)
}

///|
pub fn probability_to_logit(probability : Double) -> Double {
  logit(probability)
}

///|
pub struct LossAccumulator {
  kind : LossKind
  parameter : Double
  mut count : Int
  mut weight : Double
  mut total : Double
  mut absolute_gradient : Double
  mut maximum : Double
}

///|
pub fn LossAccumulator::new(
  kind : LossKind,
  parameter? : Double = 1.0,
) -> LossAccumulator {
  {
    kind,
    parameter,
    count: 0,
    weight: 0.0,
    total: 0.0,
    absolute_gradient: 0.0,
    maximum: 0.0,
  }
}

///|
pub fn LossAccumulator::update(
  self : LossAccumulator,
  prediction : Double,
  label : Double,
  weight? : Double = 1.0,
) -> Double {
  let safe_weight = if weight.is_nan() || weight <= 0.0 { 0.0 } else { weight }
  let value = loss_value(self.kind, prediction, label, parameter=self.parameter)
  let gradient = loss_gradient(
    self.kind,
    prediction,
    label,
    parameter=self.parameter,
  )
  self.count += 1
  self.weight += safe_weight
  self.total += safe_weight * value
  self.absolute_gradient += safe_weight * gradient.abs()
  if value > self.maximum {
    self.maximum = value
  }
  value
}

///|
pub fn LossAccumulator::kind(self : LossAccumulator) -> LossKind {
  self.kind
}

///|
pub fn LossAccumulator::count(self : LossAccumulator) -> Int {
  self.count
}

///|
pub fn LossAccumulator::weight(self : LossAccumulator) -> Double {
  self.weight
}

///|
pub fn LossAccumulator::mean(self : LossAccumulator) -> Double {
  if self.weight <= 0.0 {
    0.0
  } else {
    self.total / self.weight
  }
}

///|
pub fn LossAccumulator::mean_absolute_gradient(
  self : LossAccumulator,
) -> Double {
  if self.weight <= 0.0 {
    0.0
  } else {
    self.absolute_gradient / self.weight
  }
}

///|
pub fn LossAccumulator::maximum(self : LossAccumulator) -> Double {
  self.maximum
}

///|
pub fn LossAccumulator::reset(self : LossAccumulator) -> Unit {
  self.count = 0
  self.weight = 0.0
  self.total = 0.0
  self.absolute_gradient = 0.0
  self.maximum = 0.0
}

///|
pub struct LossSchedule {
  initial : Double
  decay : Double
  floor : Double
  mut step : Int
}

///|
pub fn LossSchedule::new(
  initial : Double,
  decay? : Double = 0.0,
  floor? : Double = 0.0,
) -> LossSchedule {
  {
    initial: if initial < 0.0 {
      0.0
    } else {
      initial
    },
    decay: if decay < 0.0 {
      0.0
    } else {
      decay
    },
    floor: if floor < 0.0 {
      0.0
    } else {
      floor
    },
    step: 0,
  }
}

///|
pub fn LossSchedule::value(self : LossSchedule) -> Double {
  let denominator = 1.0 + self.decay * self.step.to_double()
  let value = if denominator <= 0.0 {
    self.initial
  } else {
    self.initial / denominator
  }
  if value < self.floor {
    self.floor
  } else {
    value
  }
}

///|
pub fn LossSchedule::advance(self : LossSchedule, steps? : Int = 1) -> Double {
  self.step += if steps < 0 { 0 } else { steps }
  self.value()
}

///|
pub fn LossSchedule::step(self : LossSchedule) -> Int {
  self.step
}

///|
pub fn LossSchedule::reset(self : LossSchedule) -> Unit {
  self.step = 0
}

///|
pub struct GradientGuard {
  lower : Double
  upper : Double
  mut clipped : Int
  mut invalid : Int
}

///|
pub fn GradientGuard::new(
  lower? : Double = -1.0,
  upper? : Double = 1.0,
) -> GradientGuard {
  let safe_lower = if lower > upper { upper } else { lower }
  let safe_upper = if lower > upper { lower } else { upper }
  { lower: safe_lower, upper: safe_upper, clipped: 0, invalid: 0 }
}

///|
pub fn GradientGuard::apply(
  self : GradientGuard,
  gradient : Array[Double],
) -> Array[Double] {
  gradient.map(value => {
    if value.is_nan() || value.is_inf() {
      self.invalid += 1
      0.0
    } else if value < self.lower {
      self.clipped += 1
      self.lower
    } else if value > self.upper {
      self.clipped += 1
      self.upper
    } else {
      value
    }
  })
}

///|
pub fn GradientGuard::clipped(self : GradientGuard) -> Int {
  self.clipped
}

///|
pub fn GradientGuard::invalid(self : GradientGuard) -> Int {
  self.invalid
}

///|
pub fn GradientGuard::reset(self : GradientGuard) -> Unit {
  self.clipped = 0
  self.invalid = 0
}

///|
pub struct PredictionGuard {
  lower : Double
  upper : Double
  mut repaired : Int
}

///|
pub fn PredictionGuard::new(lower : Double, upper : Double) -> PredictionGuard {
  { lower, upper, repaired: 0 }
}

///|
pub fn PredictionGuard::apply(
  self : PredictionGuard,
  prediction : Double,
) -> Double {
  let value = if prediction.is_nan() || prediction.is_inf() {
    self.lower
  } else {
    prediction
  }
  if value < self.lower {
    self.repaired += 1
    self.lower
  } else if value > self.upper {
    self.repaired += 1
    self.upper
  } else {
    value
  }
}

///|
pub fn PredictionGuard::repaired(self : PredictionGuard) -> Int {
  self.repaired
}

///|
pub fn PredictionGuard::reset(self : PredictionGuard) -> Unit {
  self.repaired = 0
}

///|
pub fn weighted_loss(
  kind : LossKind,
  predictions : Array[Double],
  labels : Array[Double],
  weights? : Array[Double] = [],
) -> Double {
  let size = if predictions.length() < labels.length() {
    predictions.length()
  } else {
    labels.length()
  }
  let mut total = 0.0
  let mut denominator = 0.0
  for i in 0..<size {
    let weight = if i < weights.length() { weights[i] } else { 1.0 }
    if weight > 0.0 && !weight.is_nan() {
      total += weight * loss_value(kind, predictions[i], labels[i])
      denominator += weight
    }
  }
  if denominator <= 0.0 {
    0.0
  } else {
    total / denominator
  }
}

///|
pub fn loss_gradient_vector(
  kind : LossKind,
  predictions : Array[Double],
  labels : Array[Double],
) -> Array[Double] {
  let size = if predictions.length() < labels.length() {
    predictions.length()
  } else {
    labels.length()
  }
  Array::makei(size, i => loss_gradient(kind, predictions[i], labels[i]))
}
