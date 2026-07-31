///|
/// Online Logistic Regression using Stochastic Gradient Descent.
pub struct SGDLogisticRegression {
  weights : Array[Double]
  /// Learning rate
  alpha : Double
} derive(ToJson, FromJson)

///|
/// Create a new SGD Logistic Regression model.
pub fn SGDLogisticRegression::new(
  dim : Int,
  alpha? : Double = 0.01,
) -> SGDLogisticRegression {
  { weights: Array::make(dim, 0.0), alpha }
}

///|
/// Predict the probability (0 to 1) for a given feature vector.
pub fn SGDLogisticRegression::predict(
  self : SGDLogisticRegression,
  features : Array[Double],
) -> Double {
  let mut z = 0.0
  let dim = self.weights.length()
  for i in 0..<dim {
    z += self.weights[i] * features[i]
  }

  // Custom sigmoid
  1.0 / (1.0 + @math.exp(-z))
}

///|
/// Update the model with a single sample (features and label).
/// Label should be 0.0 or 1.0.
pub fn SGDLogisticRegression::update(
  self : SGDLogisticRegression,
  features : Array[Double],
  label : Double,
) -> Unit {
  let pred = self.predict(features)
  let error = pred - label
  let dim = self.weights.length()
  for i in 0..<dim {
    let gradient = error * features[i]
    self.weights[i] -= self.alpha * gradient
  }
}
