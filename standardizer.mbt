///|
/// A standardizer that tracks running mean and variance to standardize features incrementally
/// using Welford's online algorithm.
pub struct Standardizer {
  mut count : Double
  mean : Array[Double]
  m2 : Array[Double]
} derive(ToJson, FromJson)

///|
/// Create a new feature standardizer for `dim` features.
pub fn Standardizer::new(dim : Int) -> Standardizer {
  {
    count: 0.0,
    mean: Array::make(dim, 0.0),
    m2: Array::make(dim, 0.0),
  }
}

///|
/// Update the standardizer with a new feature vector, returning the standardized features.
pub fn Standardizer::update_and_transform(self : Standardizer, features : Array[Double]) -> Array[Double] {
  let dim = self.mean.length()
  self.count += 1.0
  let res = Array::make(dim, 0.0)
  for i in 0..<dim {
    let x = features[i]
    let delta = x - self.mean[i]
    self.mean[i] += delta / self.count
    let delta2 = x - self.mean[i]
    self.m2[i] += delta * delta2
    
    let variance = if self.count > 1.0 { self.m2[i] / (self.count - 1.0) } else { 1.0 }
    let std_dev = if variance > 0.0 { variance.sqrt() } else { 1.0 }
    res[i] = (x - self.mean[i]) / std_dev
  }
  res
}
