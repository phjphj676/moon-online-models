///|
/// Online Platt scaling for turning arbitrary scores into calibrated
/// probabilities.
pub struct OnlinePlattScaler {
  mut slope : Double
  mut intercept : Double
  learning_rate : Double
  mut steps : Int
}

///|
pub fn OnlinePlattScaler::new(
  learning_rate? : Double = 0.01,
) -> OnlinePlattScaler {
  { slope: 1.0, intercept: 0.0, learning_rate, steps: 0 }
}

///|
pub fn OnlinePlattScaler::predict(
  self : OnlinePlattScaler,
  score : Double,
) -> Double {
  sigmoid(self.slope * score + self.intercept)
}

///|
pub fn OnlinePlattScaler::update(
  self : OnlinePlattScaler,
  score : Double,
  label : Double,
) -> Unit {
  let prediction = self.predict(score)
  let error = prediction - clamp(label, 0.0, 1.0)
  self.slope -= self.learning_rate * error * score
  self.intercept -= self.learning_rate * error
  self.steps += 1
}

///|
pub fn OnlinePlattScaler::slope(self : OnlinePlattScaler) -> Double {
  self.slope
}

///|
pub fn OnlinePlattScaler::intercept(self : OnlinePlattScaler) -> Double {
  self.intercept
}

///|
pub fn OnlinePlattScaler::steps(self : OnlinePlattScaler) -> Int {
  self.steps
}

///|
pub fn OnlinePlattScaler::reset(self : OnlinePlattScaler) -> Unit {
  self.slope = 1.0
  self.intercept = 0.0
  self.steps = 0
}

///|
pub struct OnlineIsotonicCalibrator {
  scores : Array[Double]
  labels : Array[Double]
  capacity : Int
}

///|
pub fn OnlineIsotonicCalibrator::new(
  capacity? : Int = 256,
) -> OnlineIsotonicCalibrator {
  { scores: [], labels: [], capacity: if capacity < 2 { 2 } else { capacity } }
}

///|
pub fn OnlineIsotonicCalibrator::update(
  self : OnlineIsotonicCalibrator,
  score : Double,
  label : Double,
) -> Unit {
  self.scores.push(score)
  self.labels.push(clamp(label, 0.0, 1.0))
  if self.scores.length() > self.capacity {
    let _ = self.scores.remove(0)
    let _ = self.labels.remove(0)
  }
}

///|
pub fn OnlineIsotonicCalibrator::predict(
  self : OnlineIsotonicCalibrator,
  score : Double,
) -> Double {
  if self.scores.is_empty() {
    sigmoid(score)
  } else {
    let mut closest = 0
    let mut distance = (self.scores[0] - score).abs()
    for i in 1..<self.scores.length() {
      let candidate = (self.scores[i] - score).abs()
      if candidate < distance {
        closest = i
        distance = candidate
      }
    }
    self.labels[closest]
  }
}

///|
pub fn OnlineIsotonicCalibrator::size(self : OnlineIsotonicCalibrator) -> Int {
  self.scores.length()
}

///|
pub fn OnlineIsotonicCalibrator::reset(self : OnlineIsotonicCalibrator) -> Unit {
  self.scores.clear()
  self.labels.clear()
}
