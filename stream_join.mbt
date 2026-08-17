///|
/// Bounded keyed join for combining asynchronous feature and label streams.
pub struct KeyedFeatureJoin {
  capacity : Int
  pending_features : Map[String, Array[Double]]
  pending_labels : Map[String, Double]
  mut matched : Int
  mut dropped : Int
}

///|
pub fn KeyedFeatureJoin::new(capacity : Int) -> KeyedFeatureJoin {
  {
    capacity: if capacity < 0 {
      0
    } else {
      capacity
    },
    pending_features: {},
    pending_labels: {},
    matched: 0,
    dropped: 0,
  }
}

///|
pub fn KeyedFeatureJoin::put_features(
  self : KeyedFeatureJoin,
  key : String,
  features : Array[Double],
) -> Array[Double]? {
  let result = self.pending_labels
    .get(key)
    .map(_ => {
      self.pending_labels.remove(key)
      self.matched += 1
      copy_vector(features)
    })
  if result is Some(_) {
    result
  } else {
    self.pending_features[key] = copy_vector(features)
    self.trim()
    None
  }
}

///|
pub fn KeyedFeatureJoin::put_label(
  self : KeyedFeatureJoin,
  key : String,
  label : Double,
) -> Array[Double]? {
  match self.pending_features.get(key) {
    Some(features) => {
      self.pending_features.remove(key)
      self.matched += 1
      Some(features)
    }
    None => {
      self.pending_labels[key] = label
      self.trim()
      None
    }
  }
}

///|
fn KeyedFeatureJoin::trim(self : KeyedFeatureJoin) -> Unit {
  while self.pending_features.length() + self.pending_labels.length() >
        self.capacity {
    match self.pending_features.keys().to_array().get(0) {
      Some(key) => self.pending_features.remove(key)
      None =>
        match self.pending_labels.keys().to_array().get(0) {
          Some(key) => self.pending_labels.remove(key)
          None => ()
        }
    }
    self.dropped += 1
  }
}

///|
pub fn KeyedFeatureJoin::pending(self : KeyedFeatureJoin) -> Int {
  self.pending_features.length() + self.pending_labels.length()
}

///|
pub fn KeyedFeatureJoin::matched(self : KeyedFeatureJoin) -> Int {
  self.matched
}

///|
pub fn KeyedFeatureJoin::dropped(self : KeyedFeatureJoin) -> Int {
  self.dropped
}

///|
pub fn KeyedFeatureJoin::reset(self : KeyedFeatureJoin) -> Unit {
  self.pending_features.clear()
  self.pending_labels.clear()
  self.matched = 0
  self.dropped = 0
}

///|
pub struct StreamWatermark {
  mut current : Int64
  allowed_lateness : Int64
  mut late_events : Int
}

///|
pub fn StreamWatermark::new(allowed_lateness : Int64) -> StreamWatermark {
  {
    current: 0,
    allowed_lateness: if allowed_lateness < 0 {
      0
    } else {
      allowed_lateness
    },
    late_events: 0,
  }
}

///|
pub fn StreamWatermark::observe(
  self : StreamWatermark,
  timestamp : Int64,
) -> Bool {
  if timestamp + self.allowed_lateness < self.current {
    self.late_events += 1
    false
  } else {
    if timestamp > self.current {
      self.current = timestamp
    }
    true
  }
}

///|
pub fn StreamWatermark::current(self : StreamWatermark) -> Int64 {
  self.current
}

///|
pub fn StreamWatermark::late_events(self : StreamWatermark) -> Int {
  self.late_events
}

///|
pub fn StreamWatermark::reset(self : StreamWatermark) -> Unit {
  self.current = 0
  self.late_events = 0
}
