///|
/// Reusable mini-batch accumulator for callers that need deterministic batch
/// boundaries while keeping the learner online between flushes.
pub struct MiniBatchAccumulator {
  dimension : Int
  features : Array[Array[Double]]
  labels : Array[Double]
  weights : Array[Double]
  capacity : Int
}

///|
pub fn MiniBatchAccumulator::new(
  dimension : Int,
  capacity? : Int = 32,
) -> MiniBatchAccumulator {
  {
    dimension: if dimension < 0 {
      0
    } else {
      dimension
    },
    features: [],
    labels: [],
    weights: [],
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
  }
}

///|
pub fn MiniBatchAccumulator::add(
  self : MiniBatchAccumulator,
  features : Array[Double],
  label : Double,
  weight? : Double = 1.0,
) -> Bool {
  if features.length() != self.dimension {
    false
  } else {
    self.features.push(copy_vector(features))
    self.labels.push(label)
    self.weights.push(weight)
    true
  }
}

///|
pub fn MiniBatchAccumulator::ready(self : MiniBatchAccumulator) -> Bool {
  self.features.length() >= self.capacity
}

///|
pub fn MiniBatchAccumulator::size(self : MiniBatchAccumulator) -> Int {
  self.features.length()
}

///|
pub fn MiniBatchAccumulator::flush(self : MiniBatchAccumulator) -> DataBatch {
  let batch = {
    features: self.features.map(row => copy_vector(row)),
    labels: copy_vector(self.labels),
    weights: copy_vector(self.weights),
  }
  self.features.clear()
  self.labels.clear()
  self.weights.clear()
  batch
}

///|
pub fn MiniBatchAccumulator::clear(self : MiniBatchAccumulator) -> Unit {
  self.features.clear()
  self.labels.clear()
  self.weights.clear()
}

///|
pub struct BatchNormalizer {
  moments : VectorMoments
  dimension : Int
  mut batches : Int
}

///|
pub fn BatchNormalizer::new(dimension : Int) -> BatchNormalizer {
  {
    moments: VectorMoments::new(dimension),
    dimension: if dimension < 0 {
      0
    } else {
      dimension
    },
    batches: 0,
  }
}

///|
pub fn BatchNormalizer::fit(self : BatchNormalizer, batch : DataBatch) -> Unit {
  for row in batch.features() {
    self.moments.update(row)
  }
  self.batches += 1
}

///|
pub fn BatchNormalizer::transform(
  self : BatchNormalizer,
  batch : DataBatch,
) -> DataBatch {
  let features = batch.features().map(row => self.moments.standardize(row))
  DataBatch::from_arrays(features, batch.labels())
}

///|
pub fn BatchNormalizer::mean(self : BatchNormalizer) -> Array[Double] {
  self.moments.mean()
}

///|
pub fn BatchNormalizer::variance(self : BatchNormalizer) -> Array[Double] {
  self.moments.variance()
}

///|
pub fn BatchNormalizer::batches(self : BatchNormalizer) -> Int {
  self.batches
}

///|
pub fn BatchNormalizer::reset(self : BatchNormalizer) -> Unit {
  self.moments.reset()
  self.batches = 0
}

///|
pub struct EarlyStopping {
  patience : Int
  minimum_delta : Double
  mut best : Double
  mut bad_rounds : Int
  mut initialized : Bool
}

///|
pub fn EarlyStopping::new(
  patience? : Int = 5,
  minimum_delta? : Double = 1.0e-4,
) -> EarlyStopping {
  {
    patience: if patience < 1 {
      1
    } else {
      patience
    },
    minimum_delta: if minimum_delta < 0.0 {
      0.0
    } else {
      minimum_delta
    },
    best: 0.0,
    bad_rounds: 0,
    initialized: false,
  }
}

///|
pub fn EarlyStopping::observe(self : EarlyStopping, loss : Double) -> Bool {
  if !self.initialized || loss < self.best - self.minimum_delta {
    self.best = loss
    self.bad_rounds = 0
    self.initialized = true
    false
  } else {
    self.bad_rounds += 1
    self.bad_rounds >= self.patience
  }
}

///|
pub fn EarlyStopping::best(self : EarlyStopping) -> Double {
  self.best
}

///|
pub fn EarlyStopping::bad_rounds(self : EarlyStopping) -> Int {
  self.bad_rounds
}

///|
pub fn EarlyStopping::should_stop(self : EarlyStopping) -> Bool {
  self.bad_rounds >= self.patience
}

///|
pub fn EarlyStopping::reset(self : EarlyStopping) -> Unit {
  self.best = 0.0
  self.bad_rounds = 0
  self.initialized = false
}

///|
pub struct LearningCurve {
  train : MetricSeries
  validation : MetricSeries
}

///|
pub fn LearningCurve::new(capacity? : Int = 256) -> LearningCurve {
  {
    train: MetricSeries::new("train", capacity~),
    validation: MetricSeries::new("validation", capacity~),
  }
}

///|
pub fn LearningCurve::record(
  self : LearningCurve,
  train_loss : Double,
  validation_loss : Double,
) -> Unit {
  self.train.record(train_loss)
  self.validation.record(validation_loss)
}

///|
pub fn LearningCurve::train(self : LearningCurve) -> Array[Double] {
  self.train.values()
}

///|
pub fn LearningCurve::validation(self : LearningCurve) -> Array[Double] {
  self.validation.values()
}

///|
pub fn LearningCurve::generalization_gap(self : LearningCurve) -> Double {
  match (self.train.last(), self.validation.last()) {
    (Some(train), Some(validation)) => validation - train
    _ => 0.0
  }
}

///|
pub fn LearningCurve::reset(self : LearningCurve) -> Unit {
  self.train.reset()
  self.validation.reset()
}
