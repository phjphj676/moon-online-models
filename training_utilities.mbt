///|
/// Train a dense Adagrad classifier over a batch and return an auditable
/// report. The function keeps the event loop explicit for easy replay.
pub fn train_adagrad(
  model : AdagradLogisticRegression,
  batch : DataBatch,
  report : TrainingReport,
) -> TrainingReport {
  let validation = batch.validate()
  if !validation.is_valid() {
    report.record(false, 0.0)
    report
  } else {
    let features = batch.features()
    let labels = batch.labels()
    let weights = batch.weights()
    for i in 0..<batch.size() {
      let loss = model.loss(features[i], labels[i])
      model.update_weighted(features[i], labels[i], weights[i])
      report.record(true, loss)
    }
    report
  }
}

///|
pub fn train_ridge(
  model : OnlineRidgeRegression,
  batch : DataBatch,
  report : TrainingReport,
) -> TrainingReport {
  let validation = batch.validate()
  if !validation.is_valid() {
    report.record(false, 0.0)
    report
  } else {
    let features = batch.features()
    let labels = batch.labels()
    let weights = batch.weights()
    for i in 0..<batch.size() {
      let loss = model.loss(features[i], labels[i])
      model.update_weighted(features[i], labels[i], weights[i])
      report.record(true, loss)
    }
    report
  }
}

///|
pub fn evaluate_binary(
  model : AdagradLogisticRegression,
  batch : DataBatch,
) -> OnlineEvaluationSession {
  let session = OnlineEvaluationSession::new()
  let features = batch.features()
  let labels = batch.labels()
  let size = if features.length() < labels.length() {
    features.length()
  } else {
    labels.length()
  }
  for i in 0..<size {
    session.observe_binary(model.predict(features[i]), labels[i])
  }
  session
}

///|
pub fn evaluate_regression(
  model : OnlineRidgeRegression,
  batch : DataBatch,
) -> OnlineEvaluationSession {
  let session = OnlineEvaluationSession::new()
  let features = batch.features()
  let labels = batch.labels()
  let size = if features.length() < labels.length() {
    features.length()
  } else {
    labels.length()
  }
  for i in 0..<size {
    session.observe_regression(model.predict(features[i]), labels[i])
  }
  session
}

///|
pub struct HoldoutSplit {
  train : DataBatch
  holdout : DataBatch
}

///|
pub fn HoldoutSplit::new(
  batch : DataBatch,
  train_ratio? : Double = 0.8,
) -> HoldoutSplit {
  let cut = (batch.size().to_double() * clamp(train_ratio, 0.0, 1.0)).to_int()
  { train: batch.slice(0, cut), holdout: batch.slice(cut, batch.size()) }
}

///|
pub fn HoldoutSplit::train(self : HoldoutSplit) -> DataBatch {
  self.train
}

///|
pub fn HoldoutSplit::holdout(self : HoldoutSplit) -> DataBatch {
  self.holdout
}

///|
pub fn HoldoutSplit::train_size(self : HoldoutSplit) -> Int {
  self.train.size()
}

///|
pub fn HoldoutSplit::holdout_size(self : HoldoutSplit) -> Int {
  self.holdout.size()
}

///|
pub struct CrossValidationFold {
  train : DataBatch
  validation : DataBatch
  index : Int
}

///|
pub fn make_fold(
  batch : DataBatch,
  folds : Int,
  index : Int,
) -> CrossValidationFold {
  let count = if folds < 1 { 1 } else { folds }
  let safe_index = if index < 0 {
    0
  } else if index >= count {
    count - 1
  } else {
    index
  }
  let size = batch.size()
  let start = size * safe_index / count
  let end = size * (safe_index + 1) / count
  let validation = batch.slice(start, end)
  let left = batch.slice(0, start)
  let right = batch.slice(end, size)
  let train_features = left.features()
  train_features.append(right.features()[:])
  let train_labels = left.labels()
  train_labels.append(right.labels()[:])
  {
    train: DataBatch::from_arrays(train_features, train_labels),
    validation,
    index: safe_index,
  }
}

///|
pub fn CrossValidationFold::train(self : CrossValidationFold) -> DataBatch {
  self.train
}

///|
pub fn CrossValidationFold::validation(self : CrossValidationFold) -> DataBatch {
  self.validation
}

///|
pub fn CrossValidationFold::index(self : CrossValidationFold) -> Int {
  self.index
}
