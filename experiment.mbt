///|
/// Bounded metric history for dashboards and regression tests.
pub struct MetricSeries {
  name : String
  capacity : Int
  values : Array[Double]
  mut seen : Int
}

///|
pub fn MetricSeries::new(name : String, capacity? : Int = 256) -> MetricSeries {
  {
    name,
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
    values: [],
    seen: 0,
  }
}

///|
pub fn MetricSeries::record(self : MetricSeries, value : Double) -> Unit {
  self.values.push(value)
  self.seen += 1
  if self.values.length() > self.capacity {
    let _ = self.values.remove(0)
  }
}

///|
pub fn MetricSeries::name(self : MetricSeries) -> String {
  self.name
}

///|
pub fn MetricSeries::values(self : MetricSeries) -> Array[Double] {
  copy_vector(self.values)
}

///|
pub fn MetricSeries::seen(self : MetricSeries) -> Int {
  self.seen
}

///|
pub fn MetricSeries::last(self : MetricSeries) -> Double? {
  self.values.last()
}

///|
pub fn MetricSeries::mean(self : MetricSeries) -> Double {
  mean_values(self.values)
}

///|
pub fn MetricSeries::minimum(self : MetricSeries) -> Double {
  if self.values.is_empty() {
    0.0
  } else {
    self.values.fold(init=self.values[0], (current, value) => {
      if value < current {
        value
      } else {
        current
      }
    })
  }
}

///|
pub fn MetricSeries::maximum(self : MetricSeries) -> Double {
  if self.values.is_empty() {
    0.0
  } else {
    self.values.fold(init=self.values[0], (current, value) => {
      if value > current {
        value
      } else {
        current
      }
    })
  }
}

///|
pub fn MetricSeries::trend(self : MetricSeries) -> Double {
  if self.values.length() < 2 {
    0.0
  } else {
    self.values[self.values.length() - 1] - self.values[0]
  }
}

///|
pub fn MetricSeries::reset(self : MetricSeries) -> Unit {
  self.values.clear()
  self.seen = 0
}

///|
pub struct ModelHealth {
  mut accepted : Int
  mut rejected : Int
  mut prediction_errors : Int
  mut inference_failures : Int
  latency : MetricSeries
  loss : MetricSeries
}

///|
pub fn ModelHealth::new() -> ModelHealth {
  {
    accepted: 0,
    rejected: 0,
    prediction_errors: 0,
    inference_failures: 0,
    latency: MetricSeries::new("latency_ms"),
    loss: MetricSeries::new("loss"),
  }
}

///|
pub fn ModelHealth::record_sample(
  self : ModelHealth,
  accepted : Bool,
  loss : Double,
  latency_ms : Double,
) -> Unit {
  if accepted {
    self.accepted += 1
  } else {
    self.rejected += 1
  }
  self.loss.record(loss)
  self.latency.record(latency_ms)
}

///|
pub fn ModelHealth::record_error(self : ModelHealth) -> Unit {
  self.inference_failures += 1
}

///|
pub fn ModelHealth::record_prediction_error(self : ModelHealth) -> Unit {
  self.prediction_errors += 1
}

///|
pub fn ModelHealth::accepted(self : ModelHealth) -> Int {
  self.accepted
}

///|
pub fn ModelHealth::rejected(self : ModelHealth) -> Int {
  self.rejected
}

///|
pub fn ModelHealth::inference_failures(self : ModelHealth) -> Int {
  self.inference_failures
}

///|
pub fn ModelHealth::acceptance_rate(self : ModelHealth) -> Double {
  let total = self.accepted + self.rejected
  if total == 0 {
    0.0
  } else {
    self.accepted.to_double() / total.to_double()
  }
}

///|
pub fn ModelHealth::mean_loss(self : ModelHealth) -> Double {
  self.loss.mean()
}

///|
pub fn ModelHealth::p95_latency(self : ModelHealth) -> Double {
  let values = self.latency.values()
  if values.is_empty() {
    0.0
  } else {
    values.sort()
    values[(0.95 * (values.length() - 1).to_double()).to_int()]
  }
}

///|
pub fn ModelHealth::reset(self : ModelHealth) -> Unit {
  self.accepted = 0
  self.rejected = 0
  self.prediction_errors = 0
  self.inference_failures = 0
  self.latency.reset()
  self.loss.reset()
}

///|
pub struct ModelCard {
  name : String
  version : String
  task : String
  owner : String
  features : Array[String]
  guarantees : Array[String]
  notes : Array[String]
}

///|
pub fn ModelCard::new(
  name : String,
  version : String,
  task : String,
  owner : String,
) -> ModelCard {
  { name, version, task, owner, features: [], guarantees: [], notes: [] }
}

///|
pub fn ModelCard::add_feature(self : ModelCard, feature : String) -> Unit {
  self.features.push(feature)
}

///|
pub fn ModelCard::add_guarantee(self : ModelCard, guarantee : String) -> Unit {
  self.guarantees.push(guarantee)
}

///|
pub fn ModelCard::add_note(self : ModelCard, note : String) -> Unit {
  self.notes.push(note)
}

///|
pub fn ModelCard::name(self : ModelCard) -> String {
  self.name
}

///|
pub fn ModelCard::version(self : ModelCard) -> String {
  self.version
}

///|
pub fn ModelCard::task(self : ModelCard) -> String {
  self.task
}

///|
pub fn ModelCard::owner(self : ModelCard) -> String {
  self.owner
}

///|
pub fn ModelCard::features(self : ModelCard) -> Array[String] {
  self.features.map(value => value)
}

///|
pub fn ModelCard::guarantees(self : ModelCard) -> Array[String] {
  self.guarantees.map(value => value)
}

///|
pub fn ModelCard::notes(self : ModelCard) -> Array[String] {
  self.notes.map(value => value)
}
