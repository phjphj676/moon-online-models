///|
/// Stateless numeric feature engineering helpers.
pub fn polynomial_features(
  values : Array[Double],
  degree : Int,
) -> Array[Double] {
  let safe_degree = if degree < 1 { 1 } else { degree }
  let result = Array::make(0, 1.0)
  for value in values {
    let mut power = 1.0
    for _ in 1..<=safe_degree {
      power *= value
      result.push(power)
    }
  }
  result
}

///|
pub fn pairwise_interactions(
  values : Array[Double],
  include_diagonal? : Bool = false,
) -> Array[Double] {
  let result = Array::make(0, 0.0)
  for i in 0..<values.length() {
    let start = if include_diagonal { i } else { i + 1 }
    for j in start..<values.length() {
      result.push(values[i] * values[j])
    }
  }
  result
}

///|
pub fn append_interactions(
  values : Array[Double],
  include_diagonal? : Bool = false,
) -> Array[Double] {
  let result = copy_vector(values)
  result.append(pairwise_interactions(values, include_diagonal~)[:])
  result
}

///|
pub struct QuantileBinner {
  boundaries : Array[Double]
}

///|
pub fn QuantileBinner::new(boundaries : Array[Double]) -> QuantileBinner {
  let sorted = copy_vector(boundaries)
  sorted.sort()
  { boundaries: sorted }
}

///|
pub fn QuantileBinner::bin(self : QuantileBinner, value : Double) -> Int {
  let mut low = 0
  let mut high = self.boundaries.length()
  while low < high {
    let middle = low + (high - low) / 2
    if value <= self.boundaries[middle] {
      high = middle
    } else {
      low = middle + 1
    }
  }
  low
}

///|
pub fn QuantileBinner::bins(self : QuantileBinner) -> Int {
  self.boundaries.length() + 1
}

///|
pub fn QuantileBinner::one_hot(
  self : QuantileBinner,
  value : Double,
) -> Array[Double] {
  one_hot(self.bin(value), self.bins())
}

///|
pub fn QuantileBinner::boundaries(self : QuantileBinner) -> Array[Double] {
  copy_vector(self.boundaries)
}

///|
pub struct OnlineQuantileSketch {
  capacity : Int
  values : Array[Double]
  mut seen : Int
}

///|
pub fn OnlineQuantileSketch::new(capacity? : Int = 256) -> OnlineQuantileSketch {
  { capacity: if capacity < 2 { 2 } else { capacity }, values: [], seen: 0 }
}

///|
pub fn OnlineQuantileSketch::update(
  self : OnlineQuantileSketch,
  value : Double,
) -> Unit {
  self.seen += 1
  self.values.push(value)
  if self.values.length() > self.capacity {
    self.values.sort()
    let stride = self.values.length() / self.capacity
    let compressed = Array::makei(self.capacity, i => self.values[i * stride])
    self.values.clear()
    self.values.append(compressed[:])
  }
}

///|
pub fn OnlineQuantileSketch::quantile(
  self : OnlineQuantileSketch,
  probability : Double,
) -> Double? {
  if self.values.is_empty() {
    None
  } else {
    let sorted = copy_vector(self.values)
    sorted.sort()
    let index = (clamp(probability, 0.0, 1.0) *
    (sorted.length() - 1).to_double()).to_int()
    Some(sorted[index])
  }
}

///|
pub fn OnlineQuantileSketch::median(self : OnlineQuantileSketch) -> Double? {
  self.quantile(0.5)
}

///|
pub fn OnlineQuantileSketch::seen(self : OnlineQuantileSketch) -> Int {
  self.seen
}

///|
pub fn OnlineQuantileSketch::reset(self : OnlineQuantileSketch) -> Unit {
  self.values.clear()
  self.seen = 0
}

///|
pub struct TargetEncoder {
  sums : Map[String, Double]
  counts : Map[String, Double]
  mut prior : RunningMean
  smoothing : Double
}

///|
pub fn TargetEncoder::new(smoothing? : Double = 10.0) -> TargetEncoder {
  {
    sums: {},
    counts: {},
    prior: RunningMean::new(),
    smoothing: if smoothing < 0.0 {
      0.0
    } else {
      smoothing
    },
  }
}

///|
pub fn TargetEncoder::update(
  self : TargetEncoder,
  category : String,
  target : Double,
  weight? : Double = 1.0,
) -> Unit {
  self.sums.update_or_default(category, 0.0, previous => {
    previous + target * weight
  })
  self.counts.update_or_default(category, 0.0, previous => previous + weight)
  self.prior.update(target)
}

///|
pub fn TargetEncoder::encode(self : TargetEncoder, category : String) -> Double {
  let prior = self.prior.value()
  let count = self.counts.get(category).unwrap_or(0.0)
  let sum = self.sums.get(category).unwrap_or(0.0)
  if count + self.smoothing <= 0.0 {
    prior
  } else {
    (sum + self.smoothing * prior) / (count + self.smoothing)
  }
}

///|
pub fn TargetEncoder::known_categories(self : TargetEncoder) -> Int {
  self.counts.length()
}

///|
pub fn TargetEncoder::reset(self : TargetEncoder) -> Unit {
  self.sums.clear()
  self.counts.clear()
  self.prior = RunningMean::new()
}
