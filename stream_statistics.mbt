///|
/// Numerically stable scalar moments using Welford's recurrence.
pub struct RunningMoments {
  mut count : Double
  mut mean : Double
  mut m2 : Double
  mut minimum : Double
  mut maximum : Double
}

///|
pub fn RunningMoments::new() -> RunningMoments {
  { count: 0.0, mean: 0.0, m2: 0.0, minimum: 0.0, maximum: 0.0 }
}

///|
pub fn RunningMoments::update(
  self : RunningMoments,
  value : Double,
  weight? : Double = 1.0,
) -> Unit {
  if weight > 0.0 {
    let previous = self.count
    self.count += weight
    let delta = value - self.mean
    self.mean += weight * delta / self.count
    self.m2 += weight * delta * (value - self.mean)
    if previous == 0.0 || value < self.minimum {
      self.minimum = value
    }
    if previous == 0.0 || value > self.maximum {
      self.maximum = value
    }
  }
}

///|
pub fn RunningMoments::merge(
  self : RunningMoments,
  other : RunningMoments,
) -> Unit {
  if other.count > 0.0 {
    if self.count == 0.0 {
      self.count = other.count
      self.mean = other.mean
      self.m2 = other.m2
      self.minimum = other.minimum
      self.maximum = other.maximum
    } else {
      let total = self.count + other.count
      let delta = other.mean - self.mean
      self.m2 += other.m2 + delta * delta * self.count * other.count / total
      self.mean = (self.mean * self.count + other.mean * other.count) / total
      self.count = total
      if other.minimum < self.minimum {
        self.minimum = other.minimum
      }
      if other.maximum > self.maximum {
        self.maximum = other.maximum
      }
    }
  }
}

///|
pub fn RunningMoments::count(self : RunningMoments) -> Double {
  self.count
}

///|
pub fn RunningMoments::mean(self : RunningMoments) -> Double {
  self.mean
}

///|
pub fn RunningMoments::variance(self : RunningMoments) -> Double {
  if self.count <= 1.0 {
    0.0
  } else {
    self.m2 / (self.count - 1.0)
  }
}

///|
pub fn RunningMoments::population_variance(self : RunningMoments) -> Double {
  if self.count <= 0.0 {
    0.0
  } else {
    self.m2 / self.count
  }
}

///|
pub fn RunningMoments::standard_deviation(self : RunningMoments) -> Double {
  self.variance().sqrt()
}

///|
pub fn RunningMoments::minimum(self : RunningMoments) -> Double {
  self.minimum
}

///|
pub fn RunningMoments::maximum(self : RunningMoments) -> Double {
  self.maximum
}

///|
pub fn RunningMoments::range(self : RunningMoments) -> Double {
  self.maximum - self.minimum
}

///|
pub fn RunningMoments::z_score(self : RunningMoments, value : Double) -> Double {
  let deviation = self.standard_deviation()
  if deviation <= 1.0e-15 {
    0.0
  } else {
    (value - self.mean) / deviation
  }
}

///|
pub fn RunningMoments::reset(self : RunningMoments) -> Unit {
  self.count = 0.0
  self.mean = 0.0
  self.m2 = 0.0
  self.minimum = 0.0
  self.maximum = 0.0
}

///|
/// Per-feature running statistics for dense streams.
pub struct VectorMoments {
  moments : Array[RunningMoments]
}

///|
pub fn VectorMoments::new(dimension : Int) -> VectorMoments {
  {
    moments: Array::makei(if dimension < 0 { 0 } else { dimension }, _ => {
      RunningMoments::new()
    }),
  }
}

///|
pub fn VectorMoments::dimension(self : VectorMoments) -> Int {
  self.moments.length()
}

///|
pub fn VectorMoments::update(
  self : VectorMoments,
  values : Array[Double],
  weight? : Double = 1.0,
) -> Unit {
  let size = if values.length() < self.moments.length() {
    values.length()
  } else {
    self.moments.length()
  }
  for i in 0..<size {
    self.moments[i].update(values[i], weight~)
  }
}

///|
pub fn VectorMoments::mean(self : VectorMoments) -> Array[Double] {
  self.moments.map(moment => moment.mean())
}

///|
pub fn VectorMoments::variance(self : VectorMoments) -> Array[Double] {
  self.moments.map(moment => moment.population_variance())
}

///|
pub fn VectorMoments::standardize(
  self : VectorMoments,
  values : Array[Double],
) -> Array[Double] {
  Array::makei(self.moments.length(), i => {
    let deviation = self.moments[i].standard_deviation()
    if deviation <= 1.0e-15 {
      0.0
    } else {
      (values.get(i).unwrap_or(0.0) - self.moments[i].mean()) / deviation
    }
  })
}

///|
pub fn VectorMoments::moment(
  self : VectorMoments,
  index : Int,
) -> RunningMoments? {
  self.moments.get(index)
}

///|
pub fn VectorMoments::reset(self : VectorMoments) -> Unit {
  for moment in self.moments {
    moment.reset()
  }
}

///|
pub struct ExponentialMovingAverage {
  alpha : Double
  mut value : Double
  mut initialized : Bool
}

///|
pub fn ExponentialMovingAverage::new(
  alpha? : Double = 0.1,
) -> ExponentialMovingAverage {
  { alpha: clamp(alpha, 1.0e-6, 1.0), value: 0.0, initialized: false }
}

///|
pub fn ExponentialMovingAverage::update(
  self : ExponentialMovingAverage,
  value : Double,
) -> Double {
  if !self.initialized {
    self.value = value
    self.initialized = true
  } else {
    self.value = moving_average(self.value, value, self.alpha)
  }
  self.value
}

///|
pub fn ExponentialMovingAverage::value(
  self : ExponentialMovingAverage,
) -> Double {
  self.value
}

///|
pub fn ExponentialMovingAverage::initialized(
  self : ExponentialMovingAverage,
) -> Bool {
  self.initialized
}

///|
pub fn ExponentialMovingAverage::reset(self : ExponentialMovingAverage) -> Unit {
  self.value = 0.0
  self.initialized = false
}

///|
pub struct ExponentialMovingVariance {
  alpha : Double
  mut mean : Double
  mut variance : Double
  mut initialized : Bool
}

///|
pub fn ExponentialMovingVariance::new(
  alpha? : Double = 0.1,
) -> ExponentialMovingVariance {
  {
    alpha: clamp(alpha, 1.0e-6, 1.0),
    mean: 0.0,
    variance: 0.0,
    initialized: false,
  }
}

///|
pub fn ExponentialMovingVariance::update(
  self : ExponentialMovingVariance,
  value : Double,
) -> Unit {
  if !self.initialized {
    self.mean = value
    self.variance = 0.0
    self.initialized = true
  } else {
    let delta = value - self.mean
    self.mean += self.alpha * delta
    self.variance = (1.0 - self.alpha) *
      (self.variance + self.alpha * delta * delta)
  }
}

///|
pub fn ExponentialMovingVariance::mean(
  self : ExponentialMovingVariance,
) -> Double {
  self.mean
}

///|
pub fn ExponentialMovingVariance::variance(
  self : ExponentialMovingVariance,
) -> Double {
  self.variance
}

///|
pub fn ExponentialMovingVariance::standard_deviation(
  self : ExponentialMovingVariance,
) -> Double {
  self.variance.sqrt()
}

///|
pub fn ExponentialMovingVariance::z_score(
  self : ExponentialMovingVariance,
  value : Double,
) -> Double {
  let deviation = self.standard_deviation()
  if deviation <= 1.0e-15 {
    0.0
  } else {
    (value - self.mean) / deviation
  }
}

///|
pub fn ExponentialMovingVariance::initialized(
  self : ExponentialMovingVariance,
) -> Bool {
  self.initialized
}

///|
pub fn ExponentialMovingVariance::reset(
  self : ExponentialMovingVariance,
) -> Unit {
  self.mean = 0.0
  self.variance = 0.0
  self.initialized = false
}
