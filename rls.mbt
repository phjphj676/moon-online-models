///|
/// Recursive Least Squares (RLS) model for online linear regression.
pub struct RLS {
  weights : Array[Double]
  /// Inverse covariance matrix
  p : Array[Array[Double]]
  /// Forgetting factor (0 < lambda <= 1)
  lambda : Double
}

///|
/// Create a new RLS model with `dim` features.
/// `lambda` is the forgetting factor (default 1.0).
/// `alpha` is the initial ridge regression penalty (default 1.0).
pub fn RLS::new(dim : Int, lambda~ : Double = 1.0, alpha~ : Double = 1.0) -> RLS {
  let weights = Array::make(dim, 0.0)
  let p = Array::make(dim, Array::make(dim, 0.0))
  // Initialize P to alpha^-1 * I
  for i in 0..<dim {
    let row = Array::make(dim, 0.0)
    row[i] = 1.0 / alpha
    p[i] = row
  }
  { weights, p, lambda }
}

///|
/// Predict the output for a given feature vector.
pub fn RLS::predict(self : RLS, features : Array[Double]) -> Double {
  let mut pred = 0.0
  let dim = self.weights.length()
  for i in 0..<dim {
    pred += self.weights[i] * features[i]
  }
  pred
}

///|
/// Update the model with a single sample (features and label).
pub fn RLS::update(self : RLS, features : Array[Double], label : Double) -> Unit {
  let dim = self.weights.length()
  
  // Calculate gain vector K = P * x / (lambda + x^T * P * x)
  let p_x = Array::make(dim, 0.0)
  let mut x_p_x = 0.0
  for i in 0..<dim {
    let mut sum = 0.0
    for j in 0..<dim {
      sum += self.p[i][j] * features[j]
    }
    p_x[i] = sum
    x_p_x += features[i] * sum
  }
  
  let denom = self.lambda + x_p_x
  let k = Array::make(dim, 0.0)
  for i in 0..<dim {
    k[i] = p_x[i] / denom
  }
  
  // Update weights: w = w + K * (y - x^T * w)
  let pred = self.predict(features)
  let error = label - pred
  for i in 0..<dim {
    self.weights[i] += k[i] * error
  }
  
  // Update inverse covariance matrix: P = (P - K * (x^T * P))
  for i in 0..<dim {
    for j in 0..<dim {
      let delta_p = k[i] * p_x[j]
      self.p[i][j] = (self.p[i][j] - delta_p) / self.lambda
    }
  }
}
