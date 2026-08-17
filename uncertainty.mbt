///|
/// Bounded conformal residual tracker for distribution-free prediction bands.
pub struct ConformalInterval {
  residuals : Array[Double]
  capacity : Int
  confidence : Double
  mut seen : Int
}

///|
pub fn ConformalInterval::new(
  capacity? : Int = 256,
  confidence? : Double = 0.9,
) -> ConformalInterval {
  {
    residuals: [],
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
    confidence: clamp(confidence, 0.0, 1.0),
    seen: 0,
  }
}

///|
pub fn ConformalInterval::observe(
  self : ConformalInterval,
  prediction : Double,
  label : Double,
) -> Unit {
  let error = (label - prediction).abs()
  self.residuals.push(error)
  self.seen += 1
  if self.residuals.length() > self.capacity {
    let _ = self.residuals.remove(0)
  }
}

///|
pub fn ConformalInterval::radius(self : ConformalInterval) -> Double {
  if self.residuals.is_empty() {
    0.0
  } else {
    let values = copy_vector(self.residuals)
    values.sort()
    let rank = (self.confidence * values.length().to_double()).to_int()
    values[if rank >= values.length() { values.length() - 1 } else { rank }]
  }
}

///|
pub fn ConformalInterval::interval(
  self : ConformalInterval,
  prediction : Double,
) -> (Double, Double) {
  let radius = self.radius()
  (prediction - radius, prediction + radius)
}

///|
pub fn ConformalInterval::coverage(
  self : ConformalInterval,
  prediction : Double,
  label : Double,
) -> Bool {
  let interval = self.interval(prediction)
  label >= interval.0 && label <= interval.1
}

///|
pub fn ConformalInterval::size(self : ConformalInterval) -> Int {
  self.residuals.length()
}

///|
pub fn ConformalInterval::seen(self : ConformalInterval) -> Int {
  self.seen
}

///|
pub fn ConformalInterval::confidence(self : ConformalInterval) -> Double {
  self.confidence
}

///|
pub fn ConformalInterval::reset(self : ConformalInterval) -> Unit {
  self.residuals.clear()
  self.seen = 0
}

///|
pub struct OnlineQuantileInterval {
  mut lower : OnlineQuantileRegression
  mut upper : OnlineQuantileRegression
  mut observations : Int
}

///|
pub fn OnlineQuantileInterval::new(
  dimension : Int,
  coverage? : Double = 0.9,
  learning_rate? : Double = 0.01,
) -> OnlineQuantileInterval {
  let safe_coverage = clamp(coverage, 0.01, 0.99)
  {
    lower: OnlineQuantileRegression::new(
      dimension,
      quantile=(1.0 - safe_coverage) / 2.0,
      learning_rate~,
    ),
    upper: OnlineQuantileRegression::new(
      dimension,
      quantile=1.0 - (1.0 - safe_coverage) / 2.0,
      learning_rate~,
    ),
    observations: 0,
  }
}

///|
pub fn OnlineQuantileInterval::predict(
  self : OnlineQuantileInterval,
  features : Array[Double],
) -> (Double, Double) {
  let low = self.lower.predict(features)
  let high = self.upper.predict(features)
  if low <= high {
    (low, high)
  } else {
    (high, low)
  }
}

///|
pub fn OnlineQuantileInterval::update(
  self : OnlineQuantileInterval,
  features : Array[Double],
  label : Double,
) -> Unit {
  self.lower.update(features, label)
  self.upper.update(features, label)
  self.observations += 1
}

///|
pub fn OnlineQuantileInterval::contains(
  self : OnlineQuantileInterval,
  features : Array[Double],
  label : Double,
) -> Bool {
  let interval = self.predict(features)
  label >= interval.0 && label <= interval.1
}

///|
pub fn OnlineQuantileInterval::observations(
  self : OnlineQuantileInterval,
) -> Int {
  self.observations
}

///|
pub fn OnlineQuantileInterval::reset(self : OnlineQuantileInterval) -> Unit {
  self.lower = OnlineQuantileRegression::new(self.lower.weights().length())
  self.upper = OnlineQuantileRegression::new(
    self.upper.weights().length(),
    quantile=0.975,
  )
  self.observations = 0
}

///|
pub struct PredictionSet {
  labels : Array[Int]
  probabilities : Array[Double]
  threshold : Double
}

///|
pub fn PredictionSet::new(
  probabilities : Array[Double],
  threshold? : Double = 0.1,
) -> PredictionSet {
  let labels = Array::make(0, 0)
  for i in 0..<probabilities.length() {
    if probabilities[i] >= threshold {
      labels.push(i)
    }
  }
  { labels, probabilities: copy_vector(probabilities), threshold }
}

///|
pub fn PredictionSet::contains(self : PredictionSet, label : Int) -> Bool {
  self.labels.contains(label)
}

///|
pub fn PredictionSet::labels(self : PredictionSet) -> Array[Int] {
  self.labels.map(value => value)
}

///|
pub fn PredictionSet::size(self : PredictionSet) -> Int {
  self.labels.length()
}

///|
pub fn PredictionSet::probability(self : PredictionSet, label : Int) -> Double {
  self.probabilities.get(label).unwrap_or(0.0)
}
