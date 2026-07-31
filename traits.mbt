///|
/// A trait for online learning models that can be incrementally updated.
pub trait OnlineLearner {
  /// Update the model with a single sample (features and label).
  update(Self, Array[Double], Double) -> Unit

  /// Predict the output for a given feature vector.
  predict(Self, Array[Double]) -> Double
}

///|
/// A trait for models that support saving and loading state.
pub trait Snapshot {
  /// Serialize model state to Bytes
  to_bytes(Self) -> Bytes

  /// Load model state from Bytes, returning true if successful.
  from_bytes(Self, Bytes) -> Bool
}
