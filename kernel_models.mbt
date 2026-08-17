///|
/// Kernel choices for compact online non-linear models.
pub enum KernelKind {
  Linear
  Polynomial(degree~ : Int, scale~ : Double, offset~ : Double)
  Gaussian(width~ : Double)
} derive(ToJson, FromJson, Debug, Eq)

///|
pub fn KernelKind::evaluate(
  self : KernelKind,
  left : Array[Double],
  right : Array[Double],
) -> Double {
  match self {
    Linear => dot_product(left, right)
    Polynomial(degree~, scale~, offset~) =>
      @math.pow(scale * dot_product(left, right) + offset, degree.to_double())
    Gaussian(width~) => {
      let distance = kernel_squared_distance(left, right)
      let safe_width = if width <= 0.0 { 1.0 } else { width }
      @math.exp(-distance / (2.0 * safe_width * safe_width))
    }
  }
}

///|
fn kernel_squared_distance(
  left : Array[Double],
  right : Array[Double],
) -> Double {
  let size = if left.length() < right.length() {
    left.length()
  } else {
    right.length()
  }
  let mut total = 0.0
  for i in 0..<size {
    let error = left[i] - right[i]
    total += error * error
  }
  total
}

///|
pub struct KernelSupport {
  features : Array[Double]
  label : Double
  coefficient : Double
}

///|
pub fn KernelSupport::features(self : KernelSupport) -> Array[Double] {
  copy_vector(self.features)
}

///|
pub fn KernelSupport::label(self : KernelSupport) -> Double {
  self.label
}

///|
pub fn KernelSupport::coefficient(self : KernelSupport) -> Double {
  self.coefficient
}

///|
/// Budgeted kernel perceptron for non-linear binary classification.
pub struct OnlineKernelClassifier {
  kernel : KernelKind
  budget : Int
  supports : Array[KernelSupport]
  learning_rate : Double
  mut bias : Double
  mut updates : Int
}

///|
pub fn OnlineKernelClassifier::new(
  kernel? : KernelKind = Gaussian(width=1.0),
  budget? : Int = 256,
  learning_rate? : Double = 0.1,
) -> OnlineKernelClassifier {
  {
    kernel,
    budget: if budget < 0 {
      0
    } else {
      budget
    },
    supports: [],
    learning_rate,
    bias: 0.0,
    updates: 0,
  }
}

///|
pub fn OnlineKernelClassifier::support_count(
  self : OnlineKernelClassifier,
) -> Int {
  self.supports.length()
}

///|
pub fn OnlineKernelClassifier::score(
  self : OnlineKernelClassifier,
  features : Array[Double],
) -> Double {
  let mut total = self.bias
  for support in self.supports {
    total += support.coefficient *
      self.kernel.evaluate(support.features, features)
  }
  total
}

///|
pub fn OnlineKernelClassifier::predict(
  self : OnlineKernelClassifier,
  features : Array[Double],
) -> Double {
  sigmoid(self.score(features))
}

///|
pub fn OnlineKernelClassifier::predict_label(
  self : OnlineKernelClassifier,
  features : Array[Double],
) -> Double {
  if self.score(features) >= 0.0 {
    1.0
  } else {
    0.0
  }
}

///|
pub fn OnlineKernelClassifier::update(
  self : OnlineKernelClassifier,
  features : Array[Double],
  label : Double,
) -> Bool {
  let signed_label = if label >= 0.5 { 1.0 } else { -1.0 }
  let margin = signed_label * self.score(features)
  if margin > 0.0 {
    false
  } else if self.budget == 0 {
    self.bias += self.learning_rate * signed_label
    self.updates += 1
    true
  } else {
    if self.supports.length() >= self.budget {
      let _ = self.supports.remove(0)
    }
    self.supports.push({
      features: copy_vector(features),
      label: signed_label,
      coefficient: self.learning_rate * signed_label,
    })
    self.bias += self.learning_rate * signed_label
    self.updates += 1
    true
  }
}

///|
pub fn OnlineKernelClassifier::loss(
  self : OnlineKernelClassifier,
  features : Array[Double],
  label : Double,
) -> Double {
  softplus(-((if label >= 0.5 { 1.0 } else { -1.0 }) * self.score(features)))
}

///|
pub fn OnlineKernelClassifier::updates(self : OnlineKernelClassifier) -> Int {
  self.updates
}

///|
pub fn OnlineKernelClassifier::reset(self : OnlineKernelClassifier) -> Unit {
  self.supports.clear()
  self.bias = 0.0
  self.updates = 0
}

///|
/// Passive-aggressive online regression with robust step sizing.
pub struct PassiveAggressiveRegressor {
  weights : Array[Double]
  aggressiveness : Double
  epsilon : Double
  l2 : Double
  mut updates : Int
}

///|
pub fn PassiveAggressiveRegressor::new(
  dimension : Int,
  aggressiveness? : Double = 1.0,
  epsilon? : Double = 0.1,
  l2? : Double = 0.0,
) -> PassiveAggressiveRegressor {
  {
    weights: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    aggressiveness: if aggressiveness <= 0.0 {
      1.0
    } else {
      aggressiveness
    },
    epsilon: if epsilon < 0.0 {
      0.0
    } else {
      epsilon
    },
    l2,
    updates: 0,
  }
}

///|
pub fn PassiveAggressiveRegressor::predict(
  self : PassiveAggressiveRegressor,
  features : Array[Double],
) -> Double {
  dot_product(self.weights, features)
}

///|
pub fn PassiveAggressiveRegressor::update(
  self : PassiveAggressiveRegressor,
  features : Array[Double],
  label : Double,
) -> Bool {
  let residual = label - self.predict(features)
  let magnitude = if residual < 0.0 { -residual } else { residual }
  let loss = magnitude - self.epsilon
  if loss <= 0.0 {
    false
  } else {
    let norm = squared_norm(features) + self.l2
    let unconstrained = if norm <= 1.0e-15 { 0.0 } else { loss / norm }
    let step = if self.aggressiveness < unconstrained {
      self.aggressiveness
    } else {
      unconstrained
    }
    let direction = if residual < 0.0 { -1.0 } else { 1.0 }
    let limit = if features.length() < self.weights.length() {
      features.length()
    } else {
      self.weights.length()
    }
    for i in 0..<limit {
      self.weights[i] += step * direction * features[i]
    }
    self.updates += 1
    true
  }
}

///|
pub fn PassiveAggressiveRegressor::weights(
  self : PassiveAggressiveRegressor,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn PassiveAggressiveRegressor::updates(
  self : PassiveAggressiveRegressor,
) -> Int {
  self.updates
}

///|
pub fn PassiveAggressiveRegressor::reset(
  self : PassiveAggressiveRegressor,
) -> Unit {
  self.weights.fill(0.0)
  self.updates = 0
}

///|
/// Random Fourier feature map for approximating a Gaussian kernel.
pub struct RandomFourierFeatures {
  input_dimension : Int
  output_dimension : Int
  weights : Array[Array[Double]]
  phases : Array[Double]
  scale : Double
  rng : DeterministicRng
}

///|
pub fn RandomFourierFeatures::new(
  input_dimension : Int,
  output_dimension : Int,
  width? : Double = 1.0,
  seed? : UInt64 = 1,
) -> RandomFourierFeatures {
  let input = if input_dimension < 0 { 0 } else { input_dimension }
  let output = if output_dimension < 0 { 0 } else { output_dimension }
  let rng = DeterministicRng::new(seed)
  let safe_width = if width <= 0.0 { 1.0 } else { width }
  let weights = Array::makei(output, _ => {
    Array::makei(input, _ => rng.normal() / safe_width)
  })
  let phases = Array::makei(output, _ => rng.uniform(0.0, 2.0 * @math.PI))
  {
    input_dimension: input,
    output_dimension: output,
    weights,
    phases,
    scale: (2.0 / output.max(1).to_double()).sqrt(),
    rng,
  }
}

///|
pub fn RandomFourierFeatures::transform(
  self : RandomFourierFeatures,
  features : Array[Double],
) -> Array[Double] {
  Array::makei(self.output_dimension, i => {
    self.scale *
    @math.cos(dot_product(self.weights[i], features) + self.phases[i])
  })
}

///|
pub fn RandomFourierFeatures::input_dimension(
  self : RandomFourierFeatures,
) -> Int {
  self.input_dimension
}

///|
pub fn RandomFourierFeatures::output_dimension(
  self : RandomFourierFeatures,
) -> Int {
  self.output_dimension
}

///|
pub fn RandomFourierFeatures::weight_matrix(
  self : RandomFourierFeatures,
) -> Array[Array[Double]] {
  self.weights.map(row => copy_vector(row))
}
