///|
/// A reusable online feature pipeline: optional clipping, Welford
/// standardization, and deterministic sparse hashing.
pub struct FeaturePipeline {
  standardizer : Standardizer
  lower : Array[Double]
  upper : Array[Double]
  mut clip_enabled : Bool
  hasher : FeatureHasher?
  mut transformed : Int
}

///|
pub fn FeaturePipeline::new(
  dimension : Int,
  clip_lower? : Double = -1.0e12,
  clip_upper? : Double = 1.0e12,
  hashing_buckets? : Int,
) -> FeaturePipeline {
  let size = if dimension < 0 { 0 } else { dimension }
  let hasher = match hashing_buckets {
    Some(value) => Some(FeatureHasher::new(value))
    None => None
  }
  {
    standardizer: Standardizer::new(size),
    lower: Array::make(size, clip_lower),
    upper: Array::make(size, clip_upper),
    clip_enabled: clip_lower < clip_upper,
    hasher,
    transformed: 0,
  }
}

///|
pub fn FeaturePipeline::dimension(self : FeaturePipeline) -> Int {
  self.lower.length()
}

///|
pub fn FeaturePipeline::transform(
  self : FeaturePipeline,
  features : Array[Double],
) -> Array[Double] {
  let clipped = Array::makei(self.dimension(), i => {
    let value = features.get(i).unwrap_or(0.0)
    if self.clip_enabled {
      clamp(value, self.lower[i], self.upper[i])
    } else {
      value
    }
  })
  self.transformed += 1
  self.standardizer.update_and_transform(clipped)
}

///|
pub fn FeaturePipeline::fit_only(
  self : FeaturePipeline,
  features : Array[Double],
) -> Unit {
  let _ = self.transform(features)
}

///|
pub fn FeaturePipeline::transformed(self : FeaturePipeline) -> Int {
  self.transformed
}

///|
pub fn FeaturePipeline::mean(self : FeaturePipeline) -> Array[Double] {
  self.standardizer.mean()
}

///|
pub fn FeaturePipeline::variance(self : FeaturePipeline) -> Array[Double] {
  self.standardizer.variance()
}

///|
pub fn FeaturePipeline::transform_tokens(
  self : FeaturePipeline,
  tokens : Array[String],
) -> SparseVector? {
  match self.hasher {
    Some(hasher) => Some(hasher.encode(tokens))
    None => None
  }
}

///|
pub fn FeaturePipeline::set_clip(
  self : FeaturePipeline,
  lower : Double,
  upper : Double,
) -> Bool {
  if lower >= upper {
    false
  } else {
    self.lower.fill(lower)
    self.upper.fill(upper)
    self.clip_enabled = true
    true
  }
}

///|
pub fn FeaturePipeline::disable_clip(self : FeaturePipeline) -> Unit {
  self.clip_enabled = false
}

///|
pub fn FeaturePipeline::reset(self : FeaturePipeline) -> Unit {
  self.standardizer.reset()
  self.transformed = 0
}

///|
pub struct QualityGate {
  dimension : Int
  min_weight : Double
  max_weight : Double
  allow_nan_like : Bool
}

///|
pub fn QualityGate::new(
  dimension : Int,
  min_weight? : Double = 0.0,
  max_weight? : Double = 1.0e12,
) -> QualityGate {
  {
    dimension: if dimension < 0 {
      0
    } else {
      dimension
    },
    min_weight,
    max_weight: if max_weight < min_weight {
      min_weight
    } else {
      max_weight
    },
    allow_nan_like: false,
  }
}

///|
pub fn QualityGate::validate_features(
  self : QualityGate,
  features : Array[Double],
) -> ValidationReport {
  validate_vector(features, self.dimension)
}

///|
pub fn QualityGate::validate_weight(
  self : QualityGate,
  weight : Double,
) -> ValidationReport {
  if weight < self.min_weight || weight > self.max_weight {
    ValidationReport::error("sample weight outside configured range")
  } else {
    ValidationReport::ok()
  }
}

///|
pub fn QualityGate::validate_sample(
  self : QualityGate,
  features : Array[Double],
  weight : Double,
) -> ValidationReport {
  let feature_report = self.validate_features(features)
  if !feature_report.is_valid() {
    feature_report
  } else {
    self.validate_weight(weight)
  }
}

///|
pub fn QualityGate::dimension(self : QualityGate) -> Int {
  self.dimension
}

///|
pub struct RollingWindow {
  capacity : Int
  features : Array[Array[Double]]
  labels : Array[Double]
  weights : Array[Double]
  mut dropped : Int
}

///|
pub fn RollingWindow::new(capacity : Int) -> RollingWindow {
  {
    capacity: if capacity < 0 {
      0
    } else {
      capacity
    },
    features: [],
    labels: [],
    weights: [],
    dropped: 0,
  }
}

///|
pub fn RollingWindow::push(
  self : RollingWindow,
  features : Array[Double],
  label : Double,
  weight? : Double = 1.0,
) -> Bool {
  if self.capacity == 0 {
    self.dropped += 1
    false
  } else {
    self.features.push(copy_vector(features))
    self.labels.push(label)
    self.weights.push(weight)
    if self.features.length() > self.capacity {
      let _ = self.features.remove(0)
      let _ = self.labels.remove(0)
      let _ = self.weights.remove(0)
      self.dropped += 1
    }
    true
  }
}

///|
pub fn RollingWindow::size(self : RollingWindow) -> Int {
  self.features.length()
}

///|
pub fn RollingWindow::capacity(self : RollingWindow) -> Int {
  self.capacity
}

///|
pub fn RollingWindow::dropped(self : RollingWindow) -> Int {
  self.dropped
}

///|
pub fn RollingWindow::batch(self : RollingWindow) -> DataBatch {
  {
    features: self.features.map(row => copy_vector(row)),
    labels: copy_vector(self.labels),
    weights: copy_vector(self.weights),
  }
}

///|
pub fn RollingWindow::clear(self : RollingWindow) -> Unit {
  self.features.clear()
  self.labels.clear()
  self.weights.clear()
  self.dropped = 0
}

///|
pub struct EvaluationSummary {
  samples : Double
  log_loss : Double
  mse : Double
  accuracy : Double
  precision : Double
  recall : Double
  f1 : Double
  auc : Double
  ece : Double
} derive(ToJson, FromJson, Debug)

///|
pub fn EvaluationSummary::samples(self : EvaluationSummary) -> Double {
  self.samples
}

///|
pub fn EvaluationSummary::log_loss(self : EvaluationSummary) -> Double {
  self.log_loss
}

///|
pub fn EvaluationSummary::mse(self : EvaluationSummary) -> Double {
  self.mse
}

///|
pub fn EvaluationSummary::accuracy(self : EvaluationSummary) -> Double {
  self.accuracy
}

///|
pub fn EvaluationSummary::precision(self : EvaluationSummary) -> Double {
  self.precision
}

///|
pub fn EvaluationSummary::recall(self : EvaluationSummary) -> Double {
  self.recall
}

///|
pub fn EvaluationSummary::f1(self : EvaluationSummary) -> Double {
  self.f1
}

///|
pub fn EvaluationSummary::auc(self : EvaluationSummary) -> Double {
  self.auc
}

///|
pub fn EvaluationSummary::ece(self : EvaluationSummary) -> Double {
  self.ece
}

///|
pub struct OnlineEvaluationSession {
  mut metrics : MetricsTracker
  mut confusion : ConfusionMatrix
  mut auc_tracker : AucTracker
  mut calibration : CalibrationTracker
  mut regression : RegressionMetrics
}

///|
pub fn OnlineEvaluationSession::new() -> OnlineEvaluationSession {
  {
    metrics: MetricsTracker::new(),
    confusion: ConfusionMatrix::new(),
    auc_tracker: AucTracker::new(),
    calibration: CalibrationTracker::new(),
    regression: RegressionMetrics::new(),
  }
}

///|
pub fn OnlineEvaluationSession::observe_binary(
  self : OnlineEvaluationSession,
  probability : Double,
  label : Double,
) -> Unit {
  self.metrics.update(probability, label)
  self.confusion.update(probability, label)
  self.auc_tracker.update(probability, label)
  self.calibration.update(probability, label)
}

///|
pub fn OnlineEvaluationSession::observe_regression(
  self : OnlineEvaluationSession,
  prediction : Double,
  label : Double,
) -> Unit {
  self.regression.update(prediction, label)
}

///|
pub fn OnlineEvaluationSession::summary(
  self : OnlineEvaluationSession,
) -> EvaluationSummary {
  {
    samples: self.regression.count() + self.auc_tracker.size().to_double(),
    log_loss: self.metrics.log_loss(),
    mse: self.metrics.mse(),
    accuracy: self.confusion.accuracy(),
    precision: self.confusion.precision(),
    recall: self.confusion.recall(),
    f1: self.confusion.f1(),
    auc: self.auc_tracker.auc(),
    ece: self.calibration.ece(),
  }
}

///|
pub fn OnlineEvaluationSession::classification(
  self : OnlineEvaluationSession,
) -> ConfusionMatrix {
  self.confusion
}

///|
pub fn OnlineEvaluationSession::regression(
  self : OnlineEvaluationSession,
) -> RegressionMetrics {
  self.regression
}

///|
pub fn OnlineEvaluationSession::reset(self : OnlineEvaluationSession) -> Unit {
  self.metrics = MetricsTracker::new()
  self.confusion = ConfusionMatrix::new()
  self.auc_tracker = AucTracker::new()
  self.calibration = CalibrationTracker::new()
  self.regression = RegressionMetrics::new()
}

///|
pub struct TrainingReport {
  mut samples : Int
  mut accepted : Int
  mut rejected : Int
  mut total_loss : Double
  mut final_loss : Double
  mut elapsed_steps : Int
} derive(ToJson, FromJson, Debug)

///|
pub fn TrainingReport::new() -> TrainingReport {
  {
    samples: 0,
    accepted: 0,
    rejected: 0,
    total_loss: 0.0,
    final_loss: 0.0,
    elapsed_steps: 0,
  }
}

///|
pub fn TrainingReport::record(
  self : TrainingReport,
  accepted : Bool,
  loss : Double,
) -> Unit {
  self.samples += 1
  if accepted {
    self.accepted += 1
  } else {
    self.rejected += 1
  }
  self.total_loss += loss
  self.final_loss = loss
  self.elapsed_steps += 1
}

///|
pub fn TrainingReport::samples(self : TrainingReport) -> Int {
  self.samples
}

///|
pub fn TrainingReport::accepted(self : TrainingReport) -> Int {
  self.accepted
}

///|
pub fn TrainingReport::rejected(self : TrainingReport) -> Int {
  self.rejected
}

///|
pub fn TrainingReport::mean_loss(self : TrainingReport) -> Double {
  if self.accepted == 0 {
    0.0
  } else {
    self.total_loss / self.accepted.to_double()
  }
}

///|
pub fn TrainingReport::final_loss(self : TrainingReport) -> Double {
  self.final_loss
}

///|
pub fn TrainingReport::reset(self : TrainingReport) -> Unit {
  self.samples = 0
  self.accepted = 0
  self.rejected = 0
  self.total_loss = 0.0
  self.final_loss = 0.0
  self.elapsed_steps = 0
}
