///|
/// Per-class Gaussian sufficient statistics.
pub struct GaussianClassStats {
  mut count : Double
  mean : Array[Double]
  m2 : Array[Double]
}

///|
pub fn GaussianClassStats::new(dimension : Int) -> GaussianClassStats {
  {
    count: 0.0,
    mean: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    m2: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
  }
}

///|
pub fn GaussianClassStats::update(
  self : GaussianClassStats,
  features : Array[Double],
  weight? : Double = 1.0,
) -> Unit {
  if weight > 0.0 {
    let previous = self.count
    self.count += weight
    let limit = if features.length() < self.mean.length() {
      features.length()
    } else {
      self.mean.length()
    }
    for i in 0..<limit {
      let delta = features[i] - self.mean[i]
      self.mean[i] += weight * delta / self.count
      self.m2[i] += weight * delta * (features[i] - self.mean[i])
    }
    let _ = previous
  }
}

///|
pub fn GaussianClassStats::count(self : GaussianClassStats) -> Double {
  self.count
}

///|
pub fn GaussianClassStats::mean(self : GaussianClassStats) -> Array[Double] {
  copy_vector(self.mean)
}

///|
pub fn GaussianClassStats::variance(
  self : GaussianClassStats,
  smoothing? : Double = 1.0e-9,
) -> Array[Double] {
  Array::makei(self.mean.length(), i => {
    let value = if self.count <= 1.0 {
      0.0
    } else {
      self.m2[i] / (self.count - 1.0)
    }
    if value < smoothing {
      smoothing
    } else {
      value
    }
  })
}

///|
pub fn GaussianClassStats::log_likelihood(
  self : GaussianClassStats,
  features : Array[Double],
  smoothing? : Double = 1.0e-9,
) -> Double {
  let variances = self.variance(smoothing~)
  let limit = if features.length() < self.mean.length() {
    features.length()
  } else {
    self.mean.length()
  }
  let mut result = 0.0
  for i in 0..<limit {
    let variance = variances[i]
    let error = features[i] - self.mean[i]
    result += -0.5 *
      (@math.ln(2.0 * @math.PI * variance) + error * error / variance)
  }
  result
}

///|
pub fn GaussianClassStats::reset(self : GaussianClassStats) -> Unit {
  self.count = 0.0
  self.mean.fill(0.0)
  self.m2.fill(0.0)
}

///|
/// Incremental Gaussian Naive Bayes for categorical labels.
pub struct OnlineGaussianNB {
  classes : Array[GaussianClassStats]
  class_counts : Array[Double]
  smoothing : Double
  mut steps : Int
}

///|
pub fn OnlineGaussianNB::new(
  classes : Int,
  dimension : Int,
  smoothing? : Double = 1.0e-9,
) -> OnlineGaussianNB {
  let class_count = if classes < 0 { 0 } else { classes }
  {
    classes: Array::makei(class_count, _ => GaussianClassStats::new(dimension)),
    class_counts: Array::make(class_count, 0.0),
    smoothing,
    steps: 0,
  }
}

///|
pub fn OnlineGaussianNB::classes(self : OnlineGaussianNB) -> Int {
  self.classes.length()
}

///|
pub fn OnlineGaussianNB::dimension(self : OnlineGaussianNB) -> Int {
  if self.classes.is_empty() {
    0
  } else {
    self.classes[0].mean.length()
  }
}

///|
pub fn OnlineGaussianNB::update(
  self : OnlineGaussianNB,
  features : Array[Double],
  label : Int,
  weight? : Double = 1.0,
) -> Bool {
  if label < 0 || label >= self.classes.length() {
    false
  } else {
    self.classes[label].update(features, weight~)
    self.class_counts[label] += weight
    self.steps += 1
    true
  }
}

///|
pub fn OnlineGaussianNB::log_priors(self : OnlineGaussianNB) -> Array[Double] {
  let total = sum_values(self.class_counts)
  let denominator = total + self.smoothing * self.classes.length().to_double()
  Array::makei(self.classes.length(), i => {
    @math.ln((self.class_counts[i] + self.smoothing) / denominator)
  })
}

///|
pub fn OnlineGaussianNB::log_scores(
  self : OnlineGaussianNB,
  features : Array[Double],
) -> Array[Double] {
  let priors = self.log_priors()
  Array::makei(self.classes.length(), i => {
    priors[i] +
    self.classes[i].log_likelihood(features, smoothing=self.smoothing)
  })
}

///|
pub fn OnlineGaussianNB::predict_proba(
  self : OnlineGaussianNB,
  features : Array[Double],
) -> Array[Double] {
  probabilities_from_logits(self.log_scores(features))
}

///|
pub fn OnlineGaussianNB::predict(
  self : OnlineGaussianNB,
  features : Array[Double],
) -> Int? {
  argmax(self.log_scores(features))
}

///|
pub fn OnlineGaussianNB::class_counts(self : OnlineGaussianNB) -> Array[Double] {
  copy_vector(self.class_counts)
}

///|
pub fn OnlineGaussianNB::steps(self : OnlineGaussianNB) -> Int {
  self.steps
}

///|
pub fn OnlineGaussianNB::reset(self : OnlineGaussianNB) -> Unit {
  for class_stats in self.classes {
    class_stats.reset()
  }
  self.class_counts.fill(0.0)
  self.steps = 0
}

///|
/// Bernoulli Naive Bayes for binary sparse-style features.
pub struct OnlineBernoulliNB {
  ones : Array[Array[Double]]
  counts : Array[Double]
  smoothing : Double
  mut steps : Int
}

///|
pub fn OnlineBernoulliNB::new(
  classes : Int,
  dimension : Int,
  smoothing? : Double = 1.0,
) -> OnlineBernoulliNB {
  let class_count = if classes < 0 { 0 } else { classes }
  let size = if dimension < 0 { 0 } else { dimension }
  {
    ones: Array::makei(class_count, _ => Array::make(size, 0.0)),
    counts: Array::make(class_count, 0.0),
    smoothing,
    steps: 0,
  }
}

///|
pub fn OnlineBernoulliNB::update(
  self : OnlineBernoulliNB,
  features : Array[Double],
  label : Int,
  weight? : Double = 1.0,
) -> Bool {
  if label < 0 || label >= self.ones.length() {
    false
  } else {
    let row = self.ones[label]
    let limit = if features.length() < row.length() {
      features.length()
    } else {
      row.length()
    }
    for i in 0..<limit {
      row[i] += weight * clamp(features[i], 0.0, 1.0)
    }
    self.counts[label] += weight
    self.steps += 1
    true
  }
}

///|
pub fn OnlineBernoulliNB::feature_probability(
  self : OnlineBernoulliNB,
  label : Int,
  index : Int,
) -> Double {
  if label < 0 ||
    label >= self.ones.length() ||
    index < 0 ||
    index >= self.ones[label].length() {
    0.5
  } else {
    (self.ones[label][index] + self.smoothing) /
    (self.counts[label] + 2.0 * self.smoothing)
  }
}

///|
pub fn OnlineBernoulliNB::log_scores(
  self : OnlineBernoulliNB,
  features : Array[Double],
) -> Array[Double] {
  let total = sum_values(self.counts)
  let denominator = total + self.smoothing * self.counts.length().to_double()
  Array::makei(self.ones.length(), class_index => {
    let prior = @math.ln(
      (self.counts[class_index] + self.smoothing) / denominator,
    )
    let row = self.ones[class_index]
    let mut score = prior
    let limit = if features.length() < row.length() {
      features.length()
    } else {
      row.length()
    }
    for i in 0..<limit {
      let probability = self.feature_probability(class_index, i)
      if features[i] >= 0.5 {
        score += @math.ln(clamp_probability(probability))
      } else {
        score += @math.ln(clamp_probability(1.0 - probability))
      }
    }
    score
  })
}

///|
pub fn OnlineBernoulliNB::predict_proba(
  self : OnlineBernoulliNB,
  features : Array[Double],
) -> Array[Double] {
  probabilities_from_logits(self.log_scores(features))
}

///|
pub fn OnlineBernoulliNB::predict(
  self : OnlineBernoulliNB,
  features : Array[Double],
) -> Int? {
  argmax(self.log_scores(features))
}

///|
pub fn OnlineBernoulliNB::steps(self : OnlineBernoulliNB) -> Int {
  self.steps
}

///|
pub fn OnlineBernoulliNB::reset(self : OnlineBernoulliNB) -> Unit {
  for row in self.ones {
    row.fill(0.0)
  }
  self.counts.fill(0.0)
  self.steps = 0
}
