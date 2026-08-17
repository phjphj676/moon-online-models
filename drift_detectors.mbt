///|
/// Page-Hinkley change detector for gradual concept drift.
pub struct PageHinkleyDetector {
  threshold : Double
  delta : Double
  mut mean : Double
  mut cumulative : Double
  mut minimum : Double
  mut count : Double
}

///|
pub fn PageHinkleyDetector::new(
  threshold? : Double = 50.0,
  delta? : Double = 0.005,
) -> PageHinkleyDetector {
  {
    threshold: if threshold <= 0.0 {
      50.0
    } else {
      threshold
    },
    delta: if delta < 0.0 {
      0.0
    } else {
      delta
    },
    mean: 0.0,
    cumulative: 0.0,
    minimum: 0.0,
    count: 0.0,
  }
}

///|
pub fn PageHinkleyDetector::update(
  self : PageHinkleyDetector,
  value : Double,
) -> Bool {
  self.count += 1.0
  self.mean += (value - self.mean) / self.count
  self.cumulative += value - self.mean - self.delta
  if self.cumulative < self.minimum {
    self.minimum = self.cumulative
  }
  self.cumulative - self.minimum > self.threshold
}

///|
pub fn PageHinkleyDetector::mean(self : PageHinkleyDetector) -> Double {
  self.mean
}

///|
pub fn PageHinkleyDetector::count(self : PageHinkleyDetector) -> Double {
  self.count
}

///|
pub fn PageHinkleyDetector::reset(self : PageHinkleyDetector) -> Unit {
  self.mean = 0.0
  self.cumulative = 0.0
  self.minimum = 0.0
  self.count = 0.0
}

///|
/// Two-sided CUSUM detector for abrupt shifts around a target level.
pub struct CumulativeSumDetector {
  target : Double
  allowance : Double
  threshold : Double
  mut positive : Double
  mut negative : Double
}

///|
pub fn CumulativeSumDetector::new(
  target? : Double = 0.0,
  allowance? : Double = 0.5,
  threshold? : Double = 5.0,
) -> CumulativeSumDetector {
  {
    target,
    allowance: if allowance < 0.0 {
      0.0
    } else {
      allowance
    },
    threshold: if threshold <= 0.0 {
      5.0
    } else {
      threshold
    },
    positive: 0.0,
    negative: 0.0,
  }
}

///|
pub fn CumulativeSumDetector::update(
  self : CumulativeSumDetector,
  value : Double,
) -> Bool {
  let difference = value - self.target
  let positive_candidate = self.positive + difference - self.allowance
  let negative_candidate = self.negative - difference - self.allowance
  self.positive = if positive_candidate > 0.0 {
    positive_candidate
  } else {
    0.0
  }
  self.negative = if negative_candidate > 0.0 {
    negative_candidate
  } else {
    0.0
  }
  self.positive > self.threshold || self.negative > self.threshold
}

///|
pub fn CumulativeSumDetector::positive(self : CumulativeSumDetector) -> Double {
  self.positive
}

///|
pub fn CumulativeSumDetector::negative(self : CumulativeSumDetector) -> Double {
  self.negative
}

///|
pub fn CumulativeSumDetector::reset(self : CumulativeSumDetector) -> Unit {
  self.positive = 0.0
  self.negative = 0.0
}

///|
/// EWMA z-score detector. It reports an anomaly before updating its baseline,
/// which prevents a single spike from hiding itself.
pub struct EwmaAnomalyDetector {
  baseline : ExponentialMovingVariance
  threshold : Double
  mut anomalies : Int
}

///|
pub fn EwmaAnomalyDetector::new(
  alpha? : Double = 0.05,
  threshold? : Double = 3.0,
) -> EwmaAnomalyDetector {
  {
    baseline: ExponentialMovingVariance::new(alpha~),
    threshold: if threshold <= 0.0 {
      3.0
    } else {
      threshold
    },
    anomalies: 0,
  }
}

///|
pub fn EwmaAnomalyDetector::update(
  self : EwmaAnomalyDetector,
  value : Double,
) -> Bool {
  let anomaly = self.baseline.initialized() &&
    self.baseline.z_score(value).abs() >= self.threshold
  if anomaly {
    self.anomalies += 1
  }
  self.baseline.update(value)
  anomaly
}

///|
pub fn EwmaAnomalyDetector::score(
  self : EwmaAnomalyDetector,
  value : Double,
) -> Double {
  self.baseline.z_score(value).abs()
}

///|
pub fn EwmaAnomalyDetector::anomalies(self : EwmaAnomalyDetector) -> Int {
  self.anomalies
}

///|
pub fn EwmaAnomalyDetector::reset(self : EwmaAnomalyDetector) -> Unit {
  self.baseline.reset()
  self.anomalies = 0
}

///|
/// Diagonal multivariate z-score detector for low-memory edge deployments.
pub struct MultivariateAnomalyDetector {
  moments : VectorMoments
  threshold : Double
  mut anomalies : Int
}

///|
pub fn MultivariateAnomalyDetector::new(
  dimension : Int,
  threshold? : Double = 3.0,
) -> MultivariateAnomalyDetector {
  {
    moments: VectorMoments::new(dimension),
    threshold: if threshold <= 0.0 {
      3.0
    } else {
      threshold
    },
    anomalies: 0,
  }
}

///|
pub fn MultivariateAnomalyDetector::score(
  self : MultivariateAnomalyDetector,
  values : Array[Double],
) -> Double {
  let standardized = self.moments.standardize(values)
  squared_norm(standardized).sqrt()
}

///|
pub fn MultivariateAnomalyDetector::update(
  self : MultivariateAnomalyDetector,
  values : Array[Double],
) -> Bool {
  let score = self.score(values)
  let anomaly = self.moments.dimension() > 0 && score >= self.threshold
  if anomaly {
    self.anomalies += 1
  }
  self.moments.update(values)
  anomaly
}

///|
pub fn MultivariateAnomalyDetector::anomalies(
  self : MultivariateAnomalyDetector,
) -> Int {
  self.anomalies
}

///|
pub fn MultivariateAnomalyDetector::mean(
  self : MultivariateAnomalyDetector,
) -> Array[Double] {
  self.moments.mean()
}

///|
pub fn MultivariateAnomalyDetector::reset(
  self : MultivariateAnomalyDetector,
) -> Unit {
  self.moments.reset()
  self.anomalies = 0
}

///|
pub struct DriftMonitor {
  feature_detectors : Array[PageHinkleyDetector]
  target_detector : PageHinkleyDetector
  mut feature_drift_events : Int
  mut target_drift_events : Int
}

///|
pub fn DriftMonitor::new(
  dimension : Int,
  threshold? : Double = 50.0,
  delta? : Double = 0.005,
) -> DriftMonitor {
  {
    feature_detectors: Array::makei(if dimension < 0 { 0 } else { dimension }, _ => {
      PageHinkleyDetector::new(threshold~, delta~)
    }),
    target_detector: PageHinkleyDetector::new(threshold~, delta~),
    feature_drift_events: 0,
    target_drift_events: 0,
  }
}

///|
pub fn DriftMonitor::update(
  self : DriftMonitor,
  features : Array[Double],
  target? : Double,
) -> Bool {
  let feature_drift = Ref(false)
  let limit = if features.length() < self.feature_detectors.length() {
    features.length()
  } else {
    self.feature_detectors.length()
  }
  for i in 0..<limit {
    if self.feature_detectors[i].update(features[i]) {
      feature_drift.val = true
      self.feature_drift_events += 1
    }
  }
  let target_drift = match target {
    Some(value) => self.target_detector.update(value)
    None => false
  }
  if target_drift {
    self.target_drift_events += 1
  }
  feature_drift.val || target_drift
}

///|
pub fn DriftMonitor::feature_events(self : DriftMonitor) -> Int {
  self.feature_drift_events
}

///|
pub fn DriftMonitor::target_events(self : DriftMonitor) -> Int {
  self.target_drift_events
}

///|
pub fn DriftMonitor::reset(self : DriftMonitor) -> Unit {
  for detector in self.feature_detectors {
    detector.reset()
  }
  self.target_detector.reset()
  self.feature_drift_events = 0
  self.target_drift_events = 0
}
