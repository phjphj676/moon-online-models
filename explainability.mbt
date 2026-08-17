///|
/// Lightweight local explanations for linear and sparse predictions.
pub struct FeatureAttribution {
  index : Int
  contribution : Double
}

///|
pub fn FeatureAttribution::new(
  index : Int,
  contribution : Double,
) -> FeatureAttribution {
  { index, contribution }
}

///|
pub fn FeatureAttribution::index(self : FeatureAttribution) -> Int {
  self.index
}

///|
pub fn FeatureAttribution::contribution(self : FeatureAttribution) -> Double {
  self.contribution
}

///|
pub fn explain_linear(
  weights : Array[Double],
  features : Array[Double],
  top_k? : Int = 10,
) -> Array[FeatureAttribution] {
  let size = if weights.length() < features.length() {
    weights.length()
  } else {
    features.length()
  }
  let attributions = Array::makei(size, i => {
    FeatureAttribution::new(i, weights[i] * features[i])
  })
  attributions.sort_by((left, right) => {
    let left_magnitude = left.contribution.abs()
    let right_magnitude = right.contribution.abs()
    if left_magnitude > right_magnitude {
      -1
    } else if left_magnitude < right_magnitude {
      1
    } else {
      left.index - right.index
    }
  })
  let limit = if top_k < 0 {
    0
  } else if top_k > attributions.length() {
    attributions.length()
  } else {
    top_k
  }
  attributions[:limit].to_owned()
}

///|
pub fn explain_sparse(
  weights : SparseVector,
  features : SparseVector,
  top_k? : Int = 10,
) -> Array[FeatureAttribution] {
  let attributions = Array::make(0, FeatureAttribution::new(0, 0.0))
  for entry in features.entries() {
    attributions.push(
      FeatureAttribution::new(
        entry.index(),
        entry.value() * weights.get(entry.index()),
      ),
    )
  }
  attributions.sort_by((left, right) => {
    if left.contribution.abs() > right.contribution.abs() {
      -1
    } else if left.contribution.abs() < right.contribution.abs() {
      1
    } else {
      left.index - right.index
    }
  })
  let limit = if top_k < 0 {
    0
  } else if top_k > attributions.length() {
    attributions.length()
  } else {
    top_k
  }
  attributions[:limit].to_owned()
}

///|
pub fn attribution_sum(attributions : Array[FeatureAttribution]) -> Double {
  attributions.fold(init=0.0, (total, item) => total + item.contribution)
}

///|
pub struct StabilityTracker {
  previous : Array[Double]
  mut observations : Int
  mut drift : Double
}

///|
pub fn StabilityTracker::new(dimension : Int) -> StabilityTracker {
  {
    previous: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    observations: 0,
    drift: 0.0,
  }
}

///|
pub fn StabilityTracker::observe(
  self : StabilityTracker,
  values : Array[Double],
) -> Double {
  let size = if values.length() < self.previous.length() {
    values.length()
  } else {
    self.previous.length()
  }
  let mut distance = 0.0
  for i in 0..<size {
    distance += (values[i] - self.previous[i]).abs()
    self.previous[i] = values[i]
  }
  self.drift += distance
  self.observations += 1
  distance
}

///|
pub fn StabilityTracker::mean_drift(self : StabilityTracker) -> Double {
  if self.observations == 0 {
    0.0
  } else {
    self.drift / self.observations.to_double()
  }
}

///|
pub fn StabilityTracker::observations(self : StabilityTracker) -> Int {
  self.observations
}

///|
pub fn StabilityTracker::reset(self : StabilityTracker) -> Unit {
  self.previous.fill(0.0)
  self.observations = 0
  self.drift = 0.0
}
