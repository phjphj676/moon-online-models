///|
/// A streaming decision stump over one numeric feature.
pub struct OnlineDecisionStump {
  feature : Int
  mut threshold : Double
  mut left_positive : Double
  mut left_negative : Double
  mut right_positive : Double
  mut right_negative : Double
  mut updates : Int
}

///|
pub fn OnlineDecisionStump::new(
  feature : Int,
  threshold? : Double = 0.0,
) -> OnlineDecisionStump {
  {
    feature: if feature < 0 {
      0
    } else {
      feature
    },
    threshold,
    left_positive: 0.0,
    left_negative: 0.0,
    right_positive: 0.0,
    right_negative: 0.0,
    updates: 0,
  }
}

///|
pub fn OnlineDecisionStump::feature(self : OnlineDecisionStump) -> Int {
  self.feature
}

///|
pub fn OnlineDecisionStump::threshold(self : OnlineDecisionStump) -> Double {
  self.threshold
}

///|
pub fn OnlineDecisionStump::update(
  self : OnlineDecisionStump,
  features : Array[Double],
  label : Double,
  weight? : Double = 1.0,
) -> Unit {
  let value = features.get(self.feature).unwrap_or(0.0)
  let positive = label >= 0.5
  if value < self.threshold {
    if positive {
      self.left_positive += weight
    } else {
      self.left_negative += weight
    }
  } else if positive {
    self.right_positive += weight
  } else {
    self.right_negative += weight
  }
  self.updates += 1
}

///|
fn stump_probability(
  positive : Double,
  negative : Double,
  smoothing : Double,
) -> Double {
  (positive + smoothing) / (positive + negative + 2.0 * smoothing)
}

///|
pub fn OnlineDecisionStump::predict(
  self : OnlineDecisionStump,
  features : Array[Double],
  smoothing? : Double = 1.0,
) -> Double {
  let value = features.get(self.feature).unwrap_or(0.0)
  if value < self.threshold {
    stump_probability(self.left_positive, self.left_negative, smoothing)
  } else {
    stump_probability(self.right_positive, self.right_negative, smoothing)
  }
}

///|
pub fn OnlineDecisionStump::accuracy(self : OnlineDecisionStump) -> Double {
  let left_total = self.left_positive + self.left_negative
  let right_total = self.right_positive + self.right_negative
  let left_correct = if self.left_positive >= self.left_negative {
    self.left_positive
  } else {
    self.left_negative
  }
  let right_correct = if self.right_positive >= self.right_negative {
    self.right_positive
  } else {
    self.right_negative
  }
  let total = left_total + right_total
  if total <= 0.0 {
    0.0
  } else {
    (left_correct + right_correct) / total
  }
}

///|
pub fn OnlineDecisionStump::update_threshold(
  self : OnlineDecisionStump,
  threshold : Double,
) -> Unit {
  self.threshold = threshold
}

///|
pub fn OnlineDecisionStump::updates(self : OnlineDecisionStump) -> Int {
  self.updates
}

///|
pub fn OnlineDecisionStump::reset(self : OnlineDecisionStump) -> Unit {
  self.left_positive = 0.0
  self.left_negative = 0.0
  self.right_positive = 0.0
  self.right_negative = 0.0
  self.updates = 0
}

///|
/// A small random stump ensemble for bounded-memory online classification.
pub struct OnlineStumpEnsemble {
  stumps : Array[OnlineDecisionStump]
  rng : DeterministicRng
  mut updates : Int
}

///|
pub fn OnlineStumpEnsemble::new(
  stump_count : Int,
  dimension : Int,
  seed? : UInt64 = 1,
) -> OnlineStumpEnsemble {
  let count = if stump_count < 0 { 0 } else { stump_count }
  let size = if dimension < 1 { 1 } else { dimension }
  let rng = DeterministicRng::new(seed)
  let stumps = Array::makei(count, _ => {
    OnlineDecisionStump::new(
      rng.next_int(size),
      threshold=rng.uniform(-1.0, 1.0),
    )
  })
  { stumps, rng, updates: 0 }
}

///|
pub fn OnlineStumpEnsemble::size(self : OnlineStumpEnsemble) -> Int {
  self.stumps.length()
}

///|
pub fn OnlineStumpEnsemble::update(
  self : OnlineStumpEnsemble,
  features : Array[Double],
  label : Double,
  weight? : Double = 1.0,
) -> Unit {
  for stump in self.stumps {
    stump.update(features, label, weight~)
  }
  self.updates += 1
}

///|
pub fn OnlineStumpEnsemble::predict(
  self : OnlineStumpEnsemble,
  features : Array[Double],
) -> Double {
  if self.stumps.is_empty() {
    0.5
  } else {
    let mut total = 0.0
    for stump in self.stumps {
      total += stump.predict(features)
    }
    total / self.stumps.length().to_double()
  }
}

///|
pub fn OnlineStumpEnsemble::predict_label(
  self : OnlineStumpEnsemble,
  features : Array[Double],
  threshold? : Double = 0.5,
) -> Double {
  if self.predict(features) >= threshold {
    1.0
  } else {
    0.0
  }
}

///|
pub fn OnlineStumpEnsemble::accuracy(self : OnlineStumpEnsemble) -> Double {
  if self.stumps.is_empty() {
    0.0
  } else {
    let mut total = 0.0
    for stump in self.stumps {
      total += stump.accuracy()
    }
    total / self.stumps.length().to_double()
  }
}

///|
pub fn OnlineStumpEnsemble::feature_indices(
  self : OnlineStumpEnsemble,
) -> Array[Int] {
  self.stumps.map(stump => stump.feature())
}

///|
pub fn OnlineStumpEnsemble::thresholds(
  self : OnlineStumpEnsemble,
) -> Array[Double] {
  self.stumps.map(stump => stump.threshold())
}

///|
pub fn OnlineStumpEnsemble::updates(self : OnlineStumpEnsemble) -> Int {
  self.updates
}

///|
pub fn OnlineStumpEnsemble::reset(self : OnlineStumpEnsemble) -> Unit {
  for stump in self.stumps {
    stump.reset()
  }
  self.updates = 0
}

///|
pub struct GatedRegressor {
  low : OnlineRidgeRegression
  high : OnlineRidgeRegression
  boundary : Double
  mut low_count : Int
  mut high_count : Int
}

///|
pub fn GatedRegressor::new(
  dimension : Int,
  boundary? : Double = 0.0,
) -> GatedRegressor {
  {
    low: OnlineRidgeRegression::new(dimension),
    high: OnlineRidgeRegression::new(dimension),
    boundary,
    low_count: 0,
    high_count: 0,
  }
}

///|
pub fn GatedRegressor::predict(
  self : GatedRegressor,
  features : Array[Double],
  gate : Double,
) -> Double {
  if gate < self.boundary {
    self.low.predict(features)
  } else {
    self.high.predict(features)
  }
}

///|
pub fn GatedRegressor::update(
  self : GatedRegressor,
  features : Array[Double],
  gate : Double,
  label : Double,
) -> Unit {
  if gate < self.boundary {
    self.low.update(features, label)
    self.low_count += 1
  } else {
    self.high.update(features, label)
    self.high_count += 1
  }
}

///|
pub fn GatedRegressor::low_count(self : GatedRegressor) -> Int {
  self.low_count
}

///|
pub fn GatedRegressor::high_count(self : GatedRegressor) -> Int {
  self.high_count
}
