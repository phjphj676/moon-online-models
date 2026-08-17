///|
/// Online Poisson regression for count-valued event streams.
pub struct OnlinePoissonRegression {
  weights : Array[Double]
  learning_rate : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn OnlinePoissonRegression::new(
  dimension : Int,
  learning_rate? : Double = 0.01,
  l2? : Double = 0.0,
) -> OnlinePoissonRegression {
  {
    weights: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    learning_rate,
    l2,
    steps: 0,
  }
}

///|
pub fn OnlinePoissonRegression::rate(
  self : OnlinePoissonRegression,
  features : Array[Double],
) -> Double {
  @math.exp(clamp(dot_product(self.weights, features), -30.0, 30.0))
}

///|
pub fn OnlinePoissonRegression::predict(
  self : OnlinePoissonRegression,
  features : Array[Double],
) -> Double {
  self.rate(features)
}

///|
pub fn OnlinePoissonRegression::update(
  self : OnlinePoissonRegression,
  features : Array[Double],
  count : Double,
) -> Unit {
  let error = self.rate(features) - (if count < 0.0 { 0.0 } else { count })
  let limit = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  for i in 0..<limit {
    self.weights[i] -= self.learning_rate *
      (error * features[i] + self.l2 * self.weights[i])
  }
  self.steps += 1
}

///|
pub fn OnlinePoissonRegression::loss(
  self : OnlinePoissonRegression,
  features : Array[Double],
  count : Double,
) -> Double {
  let safe_count = if count < 0.0 { 0.0 } else { count }
  self.rate(features) -
  safe_count * @math.ln(self.rate(features)) +
  0.5 * self.l2 * squared_norm(self.weights)
}

///|
pub fn OnlinePoissonRegression::weights(
  self : OnlinePoissonRegression,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn OnlinePoissonRegression::steps(self : OnlinePoissonRegression) -> Int {
  self.steps
}

///|
pub fn OnlinePoissonRegression::reset(self : OnlinePoissonRegression) -> Unit {
  self.weights.fill(0.0)
  self.steps = 0
}

///|
pub struct OnlineGammaRegression {
  weights : Array[Double]
  learning_rate : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn OnlineGammaRegression::new(
  dimension : Int,
  learning_rate? : Double = 0.01,
  l2? : Double = 0.0,
) -> OnlineGammaRegression {
  {
    weights: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    learning_rate,
    l2,
    steps: 0,
  }
}

///|
pub fn OnlineGammaRegression::predict(
  self : OnlineGammaRegression,
  features : Array[Double],
) -> Double {
  @math.exp(clamp(dot_product(self.weights, features), -30.0, 30.0))
}

///|
pub fn OnlineGammaRegression::update(
  self : OnlineGammaRegression,
  features : Array[Double],
  value : Double,
) -> Unit {
  let target = if value <= 1.0e-9 { 1.0e-9 } else { value }
  let prediction = self.predict(features)
  let error = 1.0 - target / prediction
  let limit = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  for i in 0..<limit {
    self.weights[i] -= self.learning_rate *
      (error * features[i] + self.l2 * self.weights[i])
  }
  self.steps += 1
}

///|
pub fn OnlineGammaRegression::loss(
  self : OnlineGammaRegression,
  features : Array[Double],
  value : Double,
) -> Double {
  let target = if value <= 1.0e-9 { 1.0e-9 } else { value }
  let prediction = self.predict(features)
  target / prediction + @math.ln(prediction)
}

///|
pub fn OnlineGammaRegression::weights(
  self : OnlineGammaRegression,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn OnlineGammaRegression::steps(self : OnlineGammaRegression) -> Int {
  self.steps
}

///|
pub fn OnlineGammaRegression::reset(self : OnlineGammaRegression) -> Unit {
  self.weights.fill(0.0)
  self.steps = 0
}
