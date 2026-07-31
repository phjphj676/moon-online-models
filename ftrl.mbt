///|
/// FTRL-Proximal Online Logistic Regression Model.
/// Supports L1 and L2 regularization, ideal for large-scale features.
pub struct FTRL {
  /// Learning rate parameter alpha
  alpha : Double
  /// Learning rate parameter beta
  beta : Double
  /// L1 regularization parameter
  l1 : Double
  /// L2 regularization parameter
  l2 : Double
  /// Model weights (z)
  z : Array[Double]
  /// Sum of squared gradients (n)
  n : Array[Double]
} derive(ToJson, FromJson)

///|
/// Create a new FTRL model with `dim` features.
pub fn FTRL::new(
  dim : Int,
  alpha~ : Double = 0.1,
  beta~ : Double = 1.0,
  l1~ : Double = 1.0,
  l2~ : Double = 1.0
) -> FTRL {
  {
    alpha,
    beta,
    l1,
    l2,
    z: Array::make(dim, 0.0),
    n: Array::make(dim, 0.0),
  }
}

///|
/// Helper function to get the actual weight for feature `i` based on `z` and `n`.
fn FTRL::get_weight(self : FTRL, i : Int) -> Double {
  let z_i = self.z[i]
  let sign = if z_i < 0.0 { -1.0 } else { 1.0 }
  if (z_i * sign) <= self.l1 {
    0.0
  } else {
    let w_i = (sign * self.l1 - z_i) / ((self.beta + self.n[i].sqrt()) / self.alpha + self.l2)
    w_i
  }
}

///|
/// Predict the probability for a given feature vector.
pub fn FTRL::predict(self : FTRL, features : Array[Double]) -> Double {
  let mut p = 0.0
  let dim = self.z.length()
  for i in 0..<dim {
    let w_i = self.get_weight(i)
    p += w_i * features[i]
  }
  1.0 / (1.0 + @math.exp(-p))
}

///|
/// Update the model with a single sample (features and label).
/// Label should be 0.0 or 1.0.
pub fn FTRL::update(self : FTRL, features : Array[Double], label : Double) -> Unit {
  let pred = self.predict(features)
  let error = pred - label
  let dim = self.z.length()
  for i in 0..<dim {
    let g_i = error * features[i]
    let sigma = ( (self.n[i] + g_i * g_i).sqrt() - self.n[i].sqrt() ) / self.alpha
    self.z[i] += g_i - sigma * self.get_weight(i)
    self.n[i] += g_i * g_i
  }
}
