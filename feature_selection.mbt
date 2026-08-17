///|
/// Online absolute-correlation tracker for feature monitoring and pruning.
pub struct OnlineFeatureSelector {
  moments : VectorMoments
  target : RunningMoments
  cross : Array[Double]
  mut observations : Double
}

///|
pub fn OnlineFeatureSelector::new(dimension : Int) -> OnlineFeatureSelector {
  let size = if dimension < 0 { 0 } else { dimension }
  {
    moments: VectorMoments::new(size),
    target: RunningMoments::new(),
    cross: Array::make(size, 0.0),
    observations: 0.0,
  }
}

///|
pub fn OnlineFeatureSelector::dimension(self : OnlineFeatureSelector) -> Int {
  self.cross.length()
}

///|
pub fn OnlineFeatureSelector::update(
  self : OnlineFeatureSelector,
  features : Array[Double],
  target : Double,
) -> Unit {
  let old_mean = self.moments.mean()
  let old_target = self.target.mean()
  self.moments.update(features)
  self.target.update(target)
  self.observations += 1.0
  let limit = if features.length() < self.cross.length() {
    features.length()
  } else {
    self.cross.length()
  }
  for i in 0..<limit {
    self.cross[i] += (features[i] - old_mean.get(i).unwrap_or(0.0)) *
      (target - old_target)
  }
}

///|
pub fn OnlineFeatureSelector::correlations(
  self : OnlineFeatureSelector,
) -> Array[Double] {
  let feature_variance = self.moments.variance()
  let target_variance = self.target.population_variance()
  Array::makei(self.dimension(), i => {
    let denominator = (feature_variance[i] * target_variance).sqrt()
    if self.observations <= 1.0 || denominator <= 1.0e-15 {
      0.0
    } else {
      self.cross[i] / (self.observations * denominator)
    }
  })
}

///|
pub fn OnlineFeatureSelector::absolute_importance(
  self : OnlineFeatureSelector,
) -> Array[Double] {
  self.correlations().map(value => if value < 0.0 { -value } else { value })
}

///|
pub fn OnlineFeatureSelector::rank(
  self : OnlineFeatureSelector,
  top_k : Int,
) -> Array[Int] {
  let importance = self.absolute_importance()
  let order = Array::makei(importance.length(), i => i)
  order.sort_by((left, right) => {
    if importance[left] > importance[right] {
      -1
    } else if importance[left] < importance[right] {
      1
    } else {
      left - right
    }
  })
  let limit = if top_k < 0 {
    0
  } else if top_k > order.length() {
    order.length()
  } else {
    top_k
  }
  Array::makei(limit, i => order[i])
}

///|
pub fn OnlineFeatureSelector::select(
  self : OnlineFeatureSelector,
  threshold : Double,
) -> Array[Int] {
  let importance = self.absolute_importance()
  let result = Array::make(0, 0)
  for i in 0..<importance.length() {
    if importance[i] >= threshold {
      result.push(i)
    }
  }
  result
}

///|
pub fn OnlineFeatureSelector::observations(
  self : OnlineFeatureSelector,
) -> Double {
  self.observations
}

///|
pub fn OnlineFeatureSelector::reset(self : OnlineFeatureSelector) -> Unit {
  self.moments.reset()
  self.target.reset()
  self.cross.fill(0.0)
  self.observations = 0.0
}

///|
pub struct ExponentialImportance {
  scores : Array[Double]
  alpha : Double
  mut observations : Int
}

///|
pub fn ExponentialImportance::new(
  dimension : Int,
  alpha? : Double = 0.05,
) -> ExponentialImportance {
  {
    scores: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    alpha: clamp(alpha, 1.0e-6, 1.0),
    observations: 0,
  }
}

///|
pub fn ExponentialImportance::update(
  self : ExponentialImportance,
  gradients : Array[Double],
) -> Unit {
  let limit = if gradients.length() < self.scores.length() {
    gradients.length()
  } else {
    self.scores.length()
  }
  for i in 0..<limit {
    let magnitude = if gradients[i] < 0.0 {
      -gradients[i]
    } else {
      gradients[i]
    }
    self.scores[i] = moving_average(self.scores[i], magnitude, self.alpha)
  }
  self.observations += 1
}

///|
pub fn ExponentialImportance::scores(
  self : ExponentialImportance,
) -> Array[Double] {
  copy_vector(self.scores)
}

///|
pub fn ExponentialImportance::top_k(
  self : ExponentialImportance,
  k : Int,
) -> Array[Int] {
  let order = Array::makei(self.scores.length(), i => i)
  order.sort_by((left, right) => {
    if self.scores[left] > self.scores[right] {
      -1
    } else if self.scores[left] < self.scores[right] {
      1
    } else {
      left - right
    }
  })
  let limit = if k < 0 {
    0
  } else if k > order.length() {
    order.length()
  } else {
    k
  }
  Array::makei(limit, i => order[i])
}

///|
pub fn ExponentialImportance::observations(self : ExponentialImportance) -> Int {
  self.observations
}

///|
pub fn ExponentialImportance::reset(self : ExponentialImportance) -> Unit {
  self.scores.fill(0.0)
  self.observations = 0
}

///|
pub struct FeatureDriftSummary {
  changed : Array[Bool]
  scores : Array[Double]
}

///|
pub fn FeatureDriftSummary::new(
  scores : Array[Double],
  threshold : Double,
) -> FeatureDriftSummary {
  {
    changed: scores.map(value => value.abs() >= threshold),
    scores: copy_vector(scores),
  }
}

///|
pub fn FeatureDriftSummary::changed_count(self : FeatureDriftSummary) -> Int {
  self.changed.count_if(value => value)
}

///|
pub fn FeatureDriftSummary::changed(
  self : FeatureDriftSummary,
  index : Int,
) -> Bool {
  self.changed.get(index).unwrap_or(false)
}

///|
pub fn FeatureDriftSummary::scores(self : FeatureDriftSummary) -> Array[Double] {
  copy_vector(self.scores)
}

///|
/// A bounded vocabulary selector that keeps the most useful hashed features.
pub struct HashedFeatureSelector {
  dimension : Int
  importance : ExponentialImportance
  threshold : Double
}

///|
pub fn HashedFeatureSelector::new(
  dimension : Int,
  threshold? : Double = 0.0,
) -> HashedFeatureSelector {
  {
    dimension: if dimension < 0 {
      0
    } else {
      dimension
    },
    importance: ExponentialImportance::new(dimension),
    threshold,
  }
}

///|
pub fn HashedFeatureSelector::observe(
  self : HashedFeatureSelector,
  vector : SparseVector,
  gradient : Double,
) -> Unit {
  let dense = vector.to_dense()
  self.importance.update(dense.map(value => value * gradient))
}

///|
pub fn HashedFeatureSelector::selected(
  self : HashedFeatureSelector,
) -> Array[Int] {
  let scores = self.importance.scores()
  let result = Array::make(0, 0)
  for i in 0..<scores.length() {
    if scores[i] >= self.threshold {
      result.push(i)
    }
  }
  result
}

///|
pub fn HashedFeatureSelector::selected_top_k(
  self : HashedFeatureSelector,
  k : Int,
) -> Array[Int] {
  self.importance.top_k(k)
}

///|
pub fn HashedFeatureSelector::dimension(self : HashedFeatureSelector) -> Int {
  self.dimension
}

///|
pub fn HashedFeatureSelector::scores(
  self : HashedFeatureSelector,
) -> Array[Double] {
  self.importance.scores()
}

///|
pub fn HashedFeatureSelector::reset(self : HashedFeatureSelector) -> Unit {
  self.importance.reset()
}
