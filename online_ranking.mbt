///|
/// Online pairwise ranking model trained with a hinge objective.
pub struct PairwiseRanker {
  weights : Array[Double]
  learning_rate : Double
  margin : Double
  l2 : Double
  mut updates : Int
}

///|
pub fn PairwiseRanker::new(
  dimension : Int,
  learning_rate? : Double = 0.01,
  margin? : Double = 1.0,
  l2? : Double = 0.0,
) -> PairwiseRanker {
  {
    weights: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    learning_rate,
    margin: if margin <= 0.0 {
      1.0
    } else {
      margin
    },
    l2,
    updates: 0,
  }
}

///|
pub fn PairwiseRanker::score(
  self : PairwiseRanker,
  features : Array[Double],
) -> Double {
  dot_product(self.weights, features)
}

///|
pub fn PairwiseRanker::weights(self : PairwiseRanker) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn PairwiseRanker::dimension(self : PairwiseRanker) -> Int {
  self.weights.length()
}

///|
pub fn PairwiseRanker::compare(
  self : PairwiseRanker,
  left : Array[Double],
  right : Array[Double],
) -> Double {
  self.score(left) - self.score(right)
}

///|
pub fn PairwiseRanker::update(
  self : PairwiseRanker,
  positive : Array[Double],
  negative : Array[Double],
) -> Bool {
  let difference = self.compare(positive, negative)
  if difference >= self.margin {
    false
  } else {
    let limit = if positive.length() < negative.length() {
      positive.length()
    } else {
      negative.length()
    }
    let size = if limit < self.weights.length() {
      limit
    } else {
      self.weights.length()
    }
    for i in 0..<size {
      let gradient = positive[i] - negative[i]
      self.weights[i] += self.learning_rate *
        (gradient - self.l2 * self.weights[i])
    }
    self.updates += 1
    true
  }
}

///|
pub fn PairwiseRanker::loss(
  self : PairwiseRanker,
  positive : Array[Double],
  negative : Array[Double],
) -> Double {
  let margin_error = self.margin - self.compare(positive, negative)
  let hinge = if margin_error > 0.0 { margin_error } else { 0.0 }
  hinge + 0.5 * self.l2 * squared_norm(self.weights)
}

///|
pub fn PairwiseRanker::updates(self : PairwiseRanker) -> Int {
  self.updates
}

///|
pub fn PairwiseRanker::reset(self : PairwiseRanker) -> Unit {
  self.weights.fill(0.0)
  self.updates = 0
}

///|
pub struct RankingMetrics {
  mut queries : Double
  mut reciprocal_rank : Double
  mut ndcg_sum : Double
  hits : Array[Double]
}

///|
pub fn RankingMetrics::new(max_k? : Int = 10) -> RankingMetrics {
  {
    queries: 0.0,
    reciprocal_rank: 0.0,
    ndcg_sum: 0.0,
    hits: Array::make(if max_k < 1 { 1 } else { max_k }, 0.0),
  }
}

///|
pub fn RankingMetrics::observe(
  self : RankingMetrics,
  relevances : Array[Double],
) -> Unit {
  self.queries += 1.0
  let order = Array::makei(relevances.length(), i => i)
  order.sort_by((left, right) => {
    if relevances[left] > relevances[right] {
      -1
    } else if relevances[left] < relevances[right] {
      1
    } else {
      left - right
    }
  })
  let mut first_relevant = -1
  for rank in 0..<order.length() {
    if relevances[order[rank]] > 0.0 && first_relevant < 0 {
      first_relevant = rank
    }
  }
  if first_relevant >= 0 {
    self.reciprocal_rank += 1.0 / (first_relevant + 1).to_double()
  }
  let ideal = copy_vector(relevances)
  ideal.sort_by((left, right) => {
    if left > right {
      -1
    } else if left < right {
      1
    } else {
      0
    }
  })
  let mut dcg = 0.0
  let mut idcg = 0.0
  for rank in 0..<order.length() {
    let gain = @math.pow(2.0, relevances[order[rank]]) - 1.0
    let ideal_gain = @math.pow(2.0, ideal[rank]) - 1.0
    let discount = @math.ln((rank + 2).to_double())
    dcg += gain / discount
    idcg += ideal_gain / discount
    for k in 0..<self.hits.length() {
      if rank <= k && relevances[order[rank]] > 0.0 {
        self.hits[k] += 1.0
      }
    }
  }
  if idcg > 0.0 {
    self.ndcg_sum += dcg / idcg
  }
}

///|
pub fn RankingMetrics::mrr(self : RankingMetrics) -> Double {
  if self.queries <= 0.0 {
    0.0
  } else {
    self.reciprocal_rank / self.queries
  }
}

///|
pub fn RankingMetrics::ndcg(self : RankingMetrics) -> Double {
  if self.queries <= 0.0 {
    0.0
  } else {
    self.ndcg_sum / self.queries
  }
}

///|
pub fn RankingMetrics::recall_at(self : RankingMetrics, k : Int) -> Double {
  if k <= 0 || k > self.hits.length() || self.queries <= 0.0 {
    0.0
  } else {
    self.hits[k - 1] / self.queries
  }
}

///|
pub fn RankingMetrics::queries(self : RankingMetrics) -> Double {
  self.queries
}

///|
pub fn RankingMetrics::reset(self : RankingMetrics) -> Unit {
  self.queries = 0.0
  self.reciprocal_rank = 0.0
  self.ndcg_sum = 0.0
  self.hits.fill(0.0)
}

///|
pub struct ClickThroughRateTracker {
  mut impressions : Double
  mut clicks : Double
  mut predicted_sum : Double
  mut squared_calibration_error : Double
}

///|
pub fn ClickThroughRateTracker::new() -> ClickThroughRateTracker {
  {
    impressions: 0.0,
    clicks: 0.0,
    predicted_sum: 0.0,
    squared_calibration_error: 0.0,
  }
}

///|
pub fn ClickThroughRateTracker::update(
  self : ClickThroughRateTracker,
  probability : Double,
  clicked : Bool,
  weight? : Double = 1.0,
) -> Unit {
  let label = if clicked { 1.0 } else { 0.0 }
  self.impressions += weight
  self.clicks += weight * label
  self.predicted_sum += weight * probability
  self.squared_calibration_error += weight * squared_error(probability, label)
}

///|
pub fn ClickThroughRateTracker::ctr(self : ClickThroughRateTracker) -> Double {
  if self.impressions <= 0.0 {
    0.0
  } else {
    self.clicks / self.impressions
  }
}

///|
pub fn ClickThroughRateTracker::predicted_ctr(
  self : ClickThroughRateTracker,
) -> Double {
  if self.impressions <= 0.0 {
    0.0
  } else {
    self.predicted_sum / self.impressions
  }
}

///|
pub fn ClickThroughRateTracker::brier(self : ClickThroughRateTracker) -> Double {
  if self.impressions <= 0.0 {
    0.0
  } else {
    self.squared_calibration_error / self.impressions
  }
}

///|
pub fn ClickThroughRateTracker::impressions(
  self : ClickThroughRateTracker,
) -> Double {
  self.impressions
}

///|
pub fn ClickThroughRateTracker::reset(self : ClickThroughRateTracker) -> Unit {
  self.impressions = 0.0
  self.clicks = 0.0
  self.predicted_sum = 0.0
  self.squared_calibration_error = 0.0
}
