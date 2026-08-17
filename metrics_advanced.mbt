///|
/// Mutable binary classification confusion matrix for a stream.
pub struct ConfusionMatrix {
  mut true_positive : Double
  mut false_positive : Double
  mut true_negative : Double
  mut false_negative : Double
}

///|
pub fn ConfusionMatrix::new() -> ConfusionMatrix {
  {
    true_positive: 0.0,
    false_positive: 0.0,
    true_negative: 0.0,
    false_negative: 0.0,
  }
}

///|
pub fn ConfusionMatrix::update(
  self : ConfusionMatrix,
  prediction : Double,
  label : Double,
  threshold? : Double = 0.5,
) -> Unit {
  let predicted_positive = prediction >= threshold
  let actual_positive = label >= 0.5
  if predicted_positive && actual_positive {
    self.true_positive += 1.0
  } else if predicted_positive && !actual_positive {
    self.false_positive += 1.0
  } else if !predicted_positive && actual_positive {
    self.false_negative += 1.0
  } else {
    self.true_negative += 1.0
  }
}

///|
pub fn ConfusionMatrix::merge(
  self : ConfusionMatrix,
  other : ConfusionMatrix,
) -> Unit {
  self.true_positive += other.true_positive
  self.false_positive += other.false_positive
  self.true_negative += other.true_negative
  self.false_negative += other.false_negative
}

///|
pub fn ConfusionMatrix::tp(self : ConfusionMatrix) -> Double {
  self.true_positive
}

///|
pub fn ConfusionMatrix::fp(self : ConfusionMatrix) -> Double {
  self.false_positive
}

///|
pub fn ConfusionMatrix::tn(self : ConfusionMatrix) -> Double {
  self.true_negative
}

///|
pub fn ConfusionMatrix::false_negative_count(self : ConfusionMatrix) -> Double {
  self.false_negative
}

///|
pub fn ConfusionMatrix::total(self : ConfusionMatrix) -> Double {
  self.true_positive +
  self.false_positive +
  self.true_negative +
  self.false_negative
}

///|
pub fn ConfusionMatrix::accuracy(self : ConfusionMatrix) -> Double {
  let total = self.total()
  if total <= 0.0 {
    0.0
  } else {
    (self.true_positive + self.true_negative) / total
  }
}

///|
pub fn ConfusionMatrix::precision(self : ConfusionMatrix) -> Double {
  let denominator = self.true_positive + self.false_positive
  if denominator <= 0.0 {
    0.0
  } else {
    self.true_positive / denominator
  }
}

///|
pub fn ConfusionMatrix::recall(self : ConfusionMatrix) -> Double {
  let denominator = self.true_positive + self.false_negative
  if denominator <= 0.0 {
    0.0
  } else {
    self.true_positive / denominator
  }
}

///|
pub fn ConfusionMatrix::specificity(self : ConfusionMatrix) -> Double {
  let denominator = self.true_negative + self.false_positive
  if denominator <= 0.0 {
    0.0
  } else {
    self.true_negative / denominator
  }
}

///|
pub fn ConfusionMatrix::f1(self : ConfusionMatrix) -> Double {
  let precision = self.precision()
  let recall = self.recall()
  if precision + recall <= 0.0 {
    0.0
  } else {
    2.0 * precision * recall / (precision + recall)
  }
}

///|
pub fn ConfusionMatrix::balanced_accuracy(self : ConfusionMatrix) -> Double {
  0.5 * (self.recall() + self.specificity())
}

///|
pub fn ConfusionMatrix::mcc(self : ConfusionMatrix) -> Double {
  let numerator = self.true_positive * self.true_negative -
    self.false_positive * self.false_negative
  let denominator = ((self.true_positive + self.false_positive) *
  (self.true_positive + self.false_negative) *
  (self.true_negative + self.false_positive) *
  (self.true_negative + self.false_negative)).sqrt()
  if denominator <= 1.0e-15 {
    0.0
  } else {
    numerator / denominator
  }
}

///|
pub fn ConfusionMatrix::reset(self : ConfusionMatrix) -> Unit {
  self.true_positive = 0.0
  self.false_positive = 0.0
  self.true_negative = 0.0
  self.false_negative = 0.0
}

///|
/// Exact incremental AUC. The tracker stores event scores, so memory is O(n)
/// and the final computation is deterministic with tie handling.
pub struct AucTracker {
  scores : Array[Double]
  labels : Array[Bool]
}

///|
pub fn AucTracker::new() -> AucTracker {
  { scores: [], labels: [] }
}

///|
pub fn AucTracker::update(
  self : AucTracker,
  score : Double,
  label : Double,
) -> Unit {
  self.scores.push(score)
  self.labels.push(label >= 0.5)
}

///|
pub fn AucTracker::size(self : AucTracker) -> Int {
  self.scores.length()
}

///|
pub fn AucTracker::positive_count(self : AucTracker) -> Int {
  self.labels.count_if(value => value)
}

///|
pub fn AucTracker::negative_count(self : AucTracker) -> Int {
  self.size() - self.positive_count()
}

///|
pub fn AucTracker::auc(self : AucTracker) -> Double {
  let positives = self.positive_count()
  let negatives = self.negative_count()
  if positives == 0 || negatives == 0 {
    0.5
  } else {
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
    let mut rank_sum = 0.0
    let mut rank = 1.0
    for index in order {
      if self.labels[index] {
        rank_sum += rank
      }
      rank += 1.0
    }
    let positive_count = positives.to_double()
    let negative_count = negatives.to_double()
    let u = rank_sum - positive_count * (positive_count + 1.0) / 2.0
    1.0 - u / (positive_count * negative_count)
  }
}

///|
pub fn AucTracker::reset(self : AucTracker) -> Unit {
  self.scores.clear()
  self.labels.clear()
}

///|
/// Fixed-bin AUC approximation for bounded-memory deployments.
pub struct HistogramAuc {
  positive : Array[Double]
  negative : Array[Double]
  mut total_positive : Double
  mut total_negative : Double
}

///|
pub fn HistogramAuc::new(bins? : Int = 128) -> HistogramAuc {
  let size = if bins < 2 { 2 } else { bins }
  {
    positive: Array::make(size, 0.0),
    negative: Array::make(size, 0.0),
    total_positive: 0.0,
    total_negative: 0.0,
  }
}

///|
fn histogram_index(value : Double, bins : Int) -> Int {
  let clamped = clamp(value, 0.0, 1.0)
  let index = (clamped * bins.to_double()).to_int()
  if index >= bins {
    bins - 1
  } else {
    index
  }
}

///|
pub fn HistogramAuc::update(
  self : HistogramAuc,
  score : Double,
  label : Double,
  weight? : Double = 1.0,
) -> Unit {
  let index = histogram_index(score, self.positive.length())
  if label >= 0.5 {
    self.positive[index] += weight
    self.total_positive += weight
  } else {
    self.negative[index] += weight
    self.total_negative += weight
  }
}

///|
pub fn HistogramAuc::auc(self : HistogramAuc) -> Double {
  if self.total_positive <= 0.0 || self.total_negative <= 0.0 {
    0.5
  } else {
    let mut negatives_below = 0.0
    let mut wins = 0.0
    for i in 0..<self.positive.length() {
      wins += self.positive[i] * (negatives_below + 0.5 * self.negative[i])
      negatives_below += self.negative[i]
    }
    wins / (self.total_positive * self.total_negative)
  }
}

///|
pub fn HistogramAuc::bins(self : HistogramAuc) -> Int {
  self.positive.length()
}

///|
pub fn HistogramAuc::reset(self : HistogramAuc) -> Unit {
  self.positive.fill(0.0)
  self.negative.fill(0.0)
  self.total_positive = 0.0
  self.total_negative = 0.0
}

///|
pub struct CalibrationBin {
  mut count : Double
  mut predicted : Double
  mut observed : Double
}

///|
pub fn CalibrationBin::new() -> CalibrationBin {
  { count: 0.0, predicted: 0.0, observed: 0.0 }
}

///|
pub fn CalibrationBin::update(
  self : CalibrationBin,
  prediction : Double,
  label : Double,
) -> Unit {
  self.count += 1.0
  self.predicted += prediction
  self.observed += label
}

///|
pub fn CalibrationBin::count(self : CalibrationBin) -> Double {
  self.count
}

///|
pub fn CalibrationBin::mean_prediction(self : CalibrationBin) -> Double {
  if self.count <= 0.0 {
    0.0
  } else {
    self.predicted / self.count
  }
}

///|
pub fn CalibrationBin::mean_observed(self : CalibrationBin) -> Double {
  if self.count <= 0.0 {
    0.0
  } else {
    self.observed / self.count
  }
}

///|
pub struct CalibrationTracker {
  bins : Array[CalibrationBin]
  mut total : Double
  mut weighted_gap : Double
}

///|
pub fn CalibrationTracker::new(bin_count? : Int = 10) -> CalibrationTracker {
  let count = if bin_count < 2 { 2 } else { bin_count }
  {
    bins: Array::makei(count, _ => CalibrationBin::new()),
    total: 0.0,
    weighted_gap: 0.0,
  }
}

///|
pub fn CalibrationTracker::update(
  self : CalibrationTracker,
  prediction : Double,
  label : Double,
) -> Unit {
  let index = histogram_index(prediction, self.bins.length())
  let bin = self.bins[index]
  let before = bin.count
  bin.update(clamp(prediction, 0.0, 1.0), label)
  self.total += 1.0
  if before > 0.0 {
    self.weighted_gap += (bin.mean_prediction() - bin.mean_observed()).abs() /
      self.total
  }
}

///|
pub fn CalibrationTracker::ece(self : CalibrationTracker) -> Double {
  if self.total <= 0.0 {
    0.0
  } else {
    let mut total = 0.0
    for bin in self.bins {
      total += bin.count * (bin.mean_prediction() - bin.mean_observed()).abs()
    }
    total / self.total
  }
}

///|
pub fn CalibrationTracker::mce(self : CalibrationTracker) -> Double {
  let mut maximum = 0.0
  for bin in self.bins {
    let gap = (bin.mean_prediction() - bin.mean_observed()).abs()
    if gap > maximum {
      maximum = gap
    }
  }
  maximum
}

///|
pub fn CalibrationTracker::bins(
  self : CalibrationTracker,
) -> Array[CalibrationBin] {
  self.bins
}

///|
pub fn CalibrationTracker::reset(self : CalibrationTracker) -> Unit {
  for bin in self.bins {
    bin.count = 0.0
    bin.predicted = 0.0
    bin.observed = 0.0
  }
  self.total = 0.0
  self.weighted_gap = 0.0
}

///|
/// Regression metrics that remain stable under incremental updates.
pub struct RegressionMetrics {
  mut count : Double
  mut absolute_error : Double
  mut squared_error : Double
  mut label_sum : Double
  mut label_squared_sum : Double
  mut minimum_error : Double
  mut maximum_error : Double
}

///|
pub fn RegressionMetrics::new() -> RegressionMetrics {
  {
    count: 0.0,
    absolute_error: 0.0,
    squared_error: 0.0,
    label_sum: 0.0,
    label_squared_sum: 0.0,
    minimum_error: 0.0,
    maximum_error: 0.0,
  }
}

///|
pub fn RegressionMetrics::update(
  self : RegressionMetrics,
  prediction : Double,
  label : Double,
  weight? : Double = 1.0,
) -> Unit {
  let error = prediction - label
  let absolute = if error < 0.0 { -error } else { error }
  self.count += weight
  self.absolute_error += weight * absolute
  self.squared_error += weight * error * error
  self.label_sum += weight * label
  self.label_squared_sum += weight * label * label
  if self.count == weight || error < self.minimum_error {
    self.minimum_error = error
  }
  if self.count == weight || error > self.maximum_error {
    self.maximum_error = error
  }
}

///|
pub fn RegressionMetrics::count(self : RegressionMetrics) -> Double {
  self.count
}

///|
pub fn RegressionMetrics::mae(self : RegressionMetrics) -> Double {
  if self.count <= 0.0 {
    0.0
  } else {
    self.absolute_error / self.count
  }
}

///|
pub fn RegressionMetrics::mse(self : RegressionMetrics) -> Double {
  if self.count <= 0.0 {
    0.0
  } else {
    self.squared_error / self.count
  }
}

///|
pub fn RegressionMetrics::rmse(self : RegressionMetrics) -> Double {
  self.mse().sqrt()
}

///|
pub fn RegressionMetrics::r2(self : RegressionMetrics) -> Double {
  let total = self.label_squared_sum -
    self.label_sum * self.label_sum / self.count
  if self.count <= 0.0 || total <= 1.0e-15 {
    0.0
  } else {
    1.0 - self.squared_error / total
  }
}

///|
pub fn RegressionMetrics::minimum_error(self : RegressionMetrics) -> Double {
  self.minimum_error
}

///|
pub fn RegressionMetrics::maximum_error(self : RegressionMetrics) -> Double {
  self.maximum_error
}

///|
pub fn RegressionMetrics::reset(self : RegressionMetrics) -> Unit {
  self.count = 0.0
  self.absolute_error = 0.0
  self.squared_error = 0.0
  self.label_sum = 0.0
  self.label_squared_sum = 0.0
  self.minimum_error = 0.0
  self.maximum_error = 0.0
}

///|
pub struct TopKMetrics {
  mut total : Double
  hits : Array[Double]
}

///|
pub fn TopKMetrics::new(max_k? : Int = 10) -> TopKMetrics {
  { total: 0.0, hits: Array::make(if max_k < 1 { 1 } else { max_k }, 0.0) }
}

///|
pub fn TopKMetrics::update(
  self : TopKMetrics,
  ranked : Array[Int],
  label : Int,
) -> Unit {
  self.total += 1.0
  for k in 0..<self.hits.length() {
    let limit = if k + 1 < ranked.length() { k + 1 } else { ranked.length() }
    let mut found = false
    for i in 0..<limit {
      if ranked[i] == label {
        found = true
      }
    }
    if found {
      self.hits[k] += 1.0
    }
  }
}

///|
pub fn TopKMetrics::accuracy(self : TopKMetrics, k : Int) -> Double {
  if k <= 0 || k > self.hits.length() || self.total <= 0.0 {
    0.0
  } else {
    self.hits[k - 1] / self.total
  }
}

///|
pub fn TopKMetrics::total(self : TopKMetrics) -> Double {
  self.total
}

///|
pub fn TopKMetrics::reset(self : TopKMetrics) -> Unit {
  self.total = 0.0
  self.hits.fill(0.0)
}
