///|
pub enum MissingValuePolicy {
  Reject
  Zero
  Mean
  CarryForward
} derive(ToJson, FromJson, Debug, Eq)

///|
pub struct DataQualityReport {
  mut rows : Int
  mut accepted : Int
  mut rejected : Int
  mut missing_values : Int
  mut dimension_errors : Int
  mut out_of_range : Int
}

///|
pub fn DataQualityReport::new() -> DataQualityReport {
  {
    rows: 0,
    accepted: 0,
    rejected: 0,
    missing_values: 0,
    dimension_errors: 0,
    out_of_range: 0,
  }
}

///|
pub fn DataQualityReport::observe(
  self : DataQualityReport,
  accepted : Bool,
  missing : Int,
  dimension_error : Bool,
  range_error : Bool,
) -> Unit {
  self.rows += 1
  self.missing_values += missing
  if dimension_error {
    self.dimension_errors += 1
  }
  if range_error {
    self.out_of_range += 1
  }
  if accepted {
    self.accepted += 1
  } else {
    self.rejected += 1
  }
}

///|
pub fn DataQualityReport::rows(self : DataQualityReport) -> Int {
  self.rows
}

///|
pub fn DataQualityReport::accepted(self : DataQualityReport) -> Int {
  self.accepted
}

///|
pub fn DataQualityReport::rejected(self : DataQualityReport) -> Int {
  self.rejected
}

///|
pub fn DataQualityReport::missing_values(self : DataQualityReport) -> Int {
  self.missing_values
}

///|
pub fn DataQualityReport::dimension_errors(self : DataQualityReport) -> Int {
  self.dimension_errors
}

///|
pub fn DataQualityReport::out_of_range(self : DataQualityReport) -> Int {
  self.out_of_range
}

///|
pub fn DataQualityReport::acceptance_rate(self : DataQualityReport) -> Double {
  if self.rows == 0 {
    0.0
  } else {
    self.accepted.to_double() / self.rows.to_double()
  }
}

///|
pub fn DataQualityReport::reset(self : DataQualityReport) -> Unit {
  self.rows = 0
  self.accepted = 0
  self.rejected = 0
  self.missing_values = 0
  self.dimension_errors = 0
  self.out_of_range = 0
}

///|
pub struct FeatureRangeMonitor {
  lower : Array[Double]
  upper : Array[Double]
  minimum : Array[Double]
  maximum : Array[Double]
  violations : Array[Int]
}

///|
pub fn FeatureRangeMonitor::new(
  lower : Array[Double],
  upper : Array[Double],
) -> FeatureRangeMonitor {
  let size = if lower.length() < upper.length() {
    lower.length()
  } else {
    upper.length()
  }
  {
    lower: Array::makei(size, i => lower[i]),
    upper: Array::makei(size, i => upper[i]),
    minimum: Array::make(size, 0.0),
    maximum: Array::make(size, 0.0),
    violations: Array::make(size, 0),
  }
}

///|
pub fn FeatureRangeMonitor::observe(
  self : FeatureRangeMonitor,
  features : Array[Double],
) -> Bool {
  let mut valid = true
  let limit = if features.length() < self.lower.length() {
    features.length()
  } else {
    self.lower.length()
  }
  for i in 0..<limit {
    let value = features[i]
    if value < self.lower[i] || value > self.upper[i] {
      self.violations[i] += 1
      valid = false
    }
    if self.violations[i] == 0 || value < self.minimum[i] {
      self.minimum[i] = value
    }
    if self.violations[i] == 0 || value > self.maximum[i] {
      self.maximum[i] = value
    }
  }
  if features.length() != self.lower.length() {
    valid = false
  }
  valid
}

///|
pub fn FeatureRangeMonitor::violations(
  self : FeatureRangeMonitor,
) -> Array[Int] {
  self.violations.map(value => value)
}

///|
pub fn FeatureRangeMonitor::minimum(
  self : FeatureRangeMonitor,
) -> Array[Double] {
  copy_vector(self.minimum)
}

///|
pub fn FeatureRangeMonitor::maximum(
  self : FeatureRangeMonitor,
) -> Array[Double] {
  copy_vector(self.maximum)
}

///|
pub fn FeatureRangeMonitor::reset(self : FeatureRangeMonitor) -> Unit {
  self.minimum.fill(0.0)
  self.maximum.fill(0.0)
  self.violations.fill(0)
}

///|
pub struct DuplicateSignatureTracker {
  signatures : Map[String, Int]
  mut rows : Int
  mut duplicates : Int
}

///|
pub fn DuplicateSignatureTracker::new() -> DuplicateSignatureTracker {
  { signatures: {}, rows: 0, duplicates: 0 }
}

///|
pub fn DuplicateSignatureTracker::observe(
  self : DuplicateSignatureTracker,
  features : Array[Double],
) -> Bool {
  let signature = features.map(value => "\{value}").join(",")
  self.rows += 1
  if self.signatures.contains(signature) {
    self.duplicates += 1
    self.signatures[signature] += 1
    true
  } else {
    self.signatures[signature] = 1
    false
  }
}

///|
pub fn DuplicateSignatureTracker::rows(self : DuplicateSignatureTracker) -> Int {
  self.rows
}

///|
pub fn DuplicateSignatureTracker::duplicates(
  self : DuplicateSignatureTracker,
) -> Int {
  self.duplicates
}

///|
pub fn DuplicateSignatureTracker::duplicate_rate(
  self : DuplicateSignatureTracker,
) -> Double {
  if self.rows == 0 {
    0.0
  } else {
    self.duplicates.to_double() / self.rows.to_double()
  }
}

///|
pub fn DuplicateSignatureTracker::reset(
  self : DuplicateSignatureTracker,
) -> Unit {
  self.signatures.clear()
  self.rows = 0
  self.duplicates = 0
}
