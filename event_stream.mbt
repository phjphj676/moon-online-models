///|
/// Immutable training event that can be replayed for deterministic validation.
pub struct TrainingEvent {
  id : String
  timestamp : Int64
  features : Array[Double]
  label : Double
  weight : Double
} derive(ToJson, FromJson, Debug)

///|
pub fn TrainingEvent::new(
  id : String,
  timestamp : Int64,
  features : Array[Double],
  label : Double,
  weight? : Double = 1.0,
) -> TrainingEvent {
  { id, timestamp, features: copy_vector(features), label, weight }
}

///|
pub fn TrainingEvent::id(self : TrainingEvent) -> String {
  self.id
}

///|
pub fn TrainingEvent::timestamp(self : TrainingEvent) -> Int64 {
  self.timestamp
}

///|
pub fn TrainingEvent::features(self : TrainingEvent) -> Array[Double] {
  copy_vector(self.features)
}

///|
pub fn TrainingEvent::label(self : TrainingEvent) -> Double {
  self.label
}

///|
pub fn TrainingEvent::weight(self : TrainingEvent) -> Double {
  self.weight
}

///|
pub struct EventLog {
  capacity : Int
  events : Array[TrainingEvent]
  ids : Map[String, Bool]
  mut appended : Int
  mut evicted : Int
}

///|
pub fn EventLog::new(capacity : Int) -> EventLog {
  {
    capacity: if capacity < 0 {
      0
    } else {
      capacity
    },
    events: [],
    ids: {},
    appended: 0,
    evicted: 0,
  }
}

///|
pub fn EventLog::append(self : EventLog, event : TrainingEvent) -> Bool {
  if self.capacity == 0 || self.ids.contains(event.id()) {
    false
  } else {
    self.events.push(event)
    self.ids[event.id()] = true
    self.appended += 1
    if self.events.length() > self.capacity {
      let removed = self.events.remove(0)
      self.ids.remove(removed.id())
      self.evicted += 1
    }
    true
  }
}

///|
pub fn EventLog::size(self : EventLog) -> Int {
  self.events.length()
}

///|
pub fn EventLog::capacity(self : EventLog) -> Int {
  self.capacity
}

///|
pub fn EventLog::events(self : EventLog) -> Array[TrainingEvent] {
  self.events.map(event => event)
}

///|
pub fn EventLog::first_timestamp(self : EventLog) -> Int64? {
  self.events.get(0).map(event => event.timestamp())
}

///|
pub fn EventLog::last_timestamp(self : EventLog) -> Int64? {
  self.events.last().map(event => event.timestamp())
}

///|
pub fn EventLog::positive_rate(self : EventLog) -> Double {
  if self.events.is_empty() {
    0.0
  } else {
    self.events.count_if(event => event.label() >= 0.5).to_double() /
    self.events.length().to_double()
  }
}

///|
pub fn EventLog::appended(self : EventLog) -> Int {
  self.appended
}

///|
pub fn EventLog::evicted(self : EventLog) -> Int {
  self.evicted
}

///|
pub fn EventLog::replay(
  self : EventLog,
  consume : (TrainingEvent) -> Unit,
) -> Unit {
  for event in self.events {
    consume(event)
  }
}

///|
pub fn EventLog::clear(self : EventLog) -> Unit {
  self.events.clear()
  self.ids.clear()
  self.appended = 0
  self.evicted = 0
}

///|
pub struct TimeWindowAggregate {
  start : Int64
  end : Int64
  mut events : Int
  mut positives : Double
  mut weight : Double
  loss : RegressionMetrics
}

///|
pub fn TimeWindowAggregate::new(
  start : Int64,
  end : Int64,
) -> TimeWindowAggregate {
  {
    start,
    end,
    events: 0,
    positives: 0.0,
    weight: 0.0,
    loss: RegressionMetrics::new(),
  }
}

///|
pub fn TimeWindowAggregate::observe(
  self : TimeWindowAggregate,
  event : TrainingEvent,
  prediction : Double,
) -> Bool {
  if event.timestamp() < self.start || event.timestamp() >= self.end {
    false
  } else {
    self.events += 1
    self.positives += event.label() * event.weight()
    self.weight += event.weight()
    self.loss.update(prediction, event.label(), weight=event.weight())
    true
  }
}

///|
pub fn TimeWindowAggregate::events(self : TimeWindowAggregate) -> Int {
  self.events
}

///|
pub fn TimeWindowAggregate::weighted_positive_rate(
  self : TimeWindowAggregate,
) -> Double {
  if self.weight <= 0.0 {
    0.0
  } else {
    self.positives / self.weight
  }
}

///|
pub fn TimeWindowAggregate::mae(self : TimeWindowAggregate) -> Double {
  self.loss.mae()
}

///|
pub fn TimeWindowAggregate::rmse(self : TimeWindowAggregate) -> Double {
  self.loss.rmse()
}

///|
pub fn TimeWindowAggregate::contains(
  self : TimeWindowAggregate,
  timestamp : Int64,
) -> Bool {
  timestamp >= self.start && timestamp < self.end
}
