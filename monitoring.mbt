///|
/// Production-facing health and SLO primitives for continuously trained models.
pub struct MetricSnapshot {
  name : String
  value : Double
  timestamp : Int64
  samples : Int
}

///|
pub fn MetricSnapshot::new(
  name : String,
  value : Double,
  timestamp : Int64,
  samples? : Int = 0,
) -> MetricSnapshot {
  { name, value, timestamp, samples: if samples < 0 { 0 } else { samples } }
}

///|
pub fn MetricSnapshot::name(self : MetricSnapshot) -> String {
  self.name
}

///|
pub fn MetricSnapshot::value(self : MetricSnapshot) -> Double {
  self.value
}

///|
pub fn MetricSnapshot::timestamp(self : MetricSnapshot) -> Int64 {
  self.timestamp
}

///|
pub fn MetricSnapshot::samples(self : MetricSnapshot) -> Int {
  self.samples
}

///|
pub(all) enum AlertSeverity {
  Info
  Warn
  Critical
} derive(Debug, Eq)

///|
pub fn alert_severity_catalog() -> Array[AlertSeverity] {
  [Info, Warn, Critical]
}

///|
pub(all) enum AlertCondition {
  GreaterThan
  LessThan
  OutsideRange
  Stale
} derive(Debug, Eq)

///|
pub fn alert_condition_catalog() -> Array[AlertCondition] {
  [GreaterThan, LessThan, OutsideRange, Stale]
}

///|
pub struct AlertRule {
  name : String
  metric : String
  condition : AlertCondition
  lower : Double
  upper : Double
  severity : AlertSeverity
  cooldown : Int64
  mut last_alert : Int64?
}

///|
pub fn AlertRule::greater(
  name : String,
  metric : String,
  threshold : Double,
  severity? : AlertSeverity = Warn,
  cooldown? : Int64 = 0,
) -> AlertRule {
  {
    name,
    metric,
    condition: GreaterThan,
    lower: threshold,
    upper: 0.0,
    severity,
    cooldown,
    last_alert: None,
  }
}

///|
pub fn AlertRule::less(
  name : String,
  metric : String,
  threshold : Double,
  severity? : AlertSeverity = Warn,
  cooldown? : Int64 = 0,
) -> AlertRule {
  {
    name,
    metric,
    condition: LessThan,
    lower: threshold,
    upper: 0.0,
    severity,
    cooldown,
    last_alert: None,
  }
}

///|
pub fn AlertRule::outside(
  name : String,
  metric : String,
  lower : Double,
  upper : Double,
  severity? : AlertSeverity = Warn,
  cooldown? : Int64 = 0,
) -> AlertRule {
  {
    name,
    metric,
    condition: OutsideRange,
    lower,
    upper,
    severity,
    cooldown,
    last_alert: None,
  }
}

///|
pub fn AlertRule::stale(
  name : String,
  metric : String,
  max_age : Int64,
  severity? : AlertSeverity = Warn,
) -> AlertRule {
  {
    name,
    metric,
    condition: Stale,
    lower: max_age.to_double(),
    upper: 0.0,
    severity,
    cooldown: 0,
    last_alert: None,
  }
}

///|
pub fn AlertRule::name(self : AlertRule) -> String {
  self.name
}

///|
pub fn AlertRule::metric(self : AlertRule) -> String {
  self.metric
}

///|
pub fn AlertRule::severity(self : AlertRule) -> AlertSeverity {
  self.severity
}

///|
pub fn AlertRule::evaluate(
  self : AlertRule,
  snapshot : MetricSnapshot,
  now : Int64,
) -> Bool {
  if snapshot.name() != self.metric {
    false
  } else {
    let triggered = match self.condition {
      GreaterThan => snapshot.value() > self.lower
      LessThan => snapshot.value() < self.lower
      OutsideRange =>
        snapshot.value() < self.lower || snapshot.value() > self.upper
      Stale => now - snapshot.timestamp() > self.lower.to_int64()
    }
    let allowed = match self.last_alert {
      None => true
      Some(previous) => now - previous >= self.cooldown
    }
    if triggered && allowed {
      self.last_alert = Some(now)
      true
    } else {
      false
    }
  }
}

///|
pub fn AlertRule::reset(self : AlertRule) -> Unit {
  self.last_alert = None
}

///|
pub struct LatencyTracker {
  samples : SequenceWindow
  mut total : Int
  mut rejected : Int
}

///|
pub fn LatencyTracker::new(capacity? : Int = 512) -> LatencyTracker {
  { samples: SequenceWindow::new(capacity), total: 0, rejected: 0 }
}

///|
pub fn LatencyTracker::observe(
  self : LatencyTracker,
  milliseconds : Double,
) -> Bool {
  self.total += 1
  if milliseconds.is_nan() || milliseconds.is_inf() || milliseconds < 0.0 {
    self.rejected += 1
    false
  } else {
    self.samples.push(milliseconds)
    true
  }
}

///|
pub fn LatencyTracker::count(self : LatencyTracker) -> Int {
  self.samples.size()
}

///|
pub fn LatencyTracker::mean(self : LatencyTracker) -> Double {
  self.samples.mean()
}

///|
pub fn LatencyTracker::p95(self : LatencyTracker) -> Double {
  self.percentile(0.95)
}

///|
pub fn LatencyTracker::percentile(
  self : LatencyTracker,
  probability : Double,
) -> Double {
  let values = self.samples.values()
  if values.is_empty() {
    0.0
  } else {
    values.sort()
    let index = (clamp(probability, 0.0, 1.0) *
    (values.length() - 1).to_double()).to_int()
    values[index]
  }
}

///|
pub fn LatencyTracker::total(self : LatencyTracker) -> Int {
  self.total
}

///|
pub fn LatencyTracker::rejected(self : LatencyTracker) -> Int {
  self.rejected
}

///|
pub fn LatencyTracker::reset(self : LatencyTracker) -> Unit {
  self.samples.clear()
  self.total = 0
  self.rejected = 0
}

///|
pub struct ErrorBudget {
  target : Double
  window : Int
  mut successes : Int
  mut failures : Int
}

///|
pub fn ErrorBudget::new(
  target? : Double = 0.99,
  window? : Int = 1000,
) -> ErrorBudget {
  {
    target: clamp(target, 0.0, 1.0),
    window: if window < 1 {
      1
    } else {
      window
    },
    successes: 0,
    failures: 0,
  }
}

///|
pub fn ErrorBudget::observe(self : ErrorBudget, success : Bool) -> Unit {
  if success {
    self.successes += 1
  } else {
    self.failures += 1
  }
  while self.successes + self.failures > self.window {
    if self.failures > 0 {
      self.failures -= 1
    } else {
      self.successes -= 1
    }
  }
}

///|
pub fn ErrorBudget::availability(self : ErrorBudget) -> Double {
  if self.successes + self.failures == 0 {
    1.0
  } else {
    self.successes.to_double() / (self.successes + self.failures).to_double()
  }
}

///|
pub fn ErrorBudget::target(self : ErrorBudget) -> Double {
  self.target
}

///|
pub fn ErrorBudget::remaining(self : ErrorBudget) -> Double {
  let budget = 1.0 - self.target
  let used = 1.0 - self.availability()
  if used >= budget {
    0.0
  } else {
    budget - used
  }
}

///|
pub fn ErrorBudget::healthy(self : ErrorBudget) -> Bool {
  self.availability() >= self.target
}

///|
pub fn ErrorBudget::successes(self : ErrorBudget) -> Int {
  self.successes
}

///|
pub fn ErrorBudget::failures(self : ErrorBudget) -> Int {
  self.failures
}

///|
pub fn ErrorBudget::reset(self : ErrorBudget) -> Unit {
  self.successes = 0
  self.failures = 0
}

///|
pub struct ModelMonitor {
  name : String
  metrics : Map[String, RunningMoments]
  rules : Array[AlertRule]
  mut snapshots : Int
  mut alerts : Int
}

///|
pub fn ModelMonitor::new(name : String) -> ModelMonitor {
  { name, metrics: {}, rules: [], snapshots: 0, alerts: 0 }
}

///|
pub fn ModelMonitor::add_rule(self : ModelMonitor, rule : AlertRule) -> Unit {
  self.rules.push(rule)
}

///|
pub fn ModelMonitor::observe(
  self : ModelMonitor,
  snapshot : MetricSnapshot,
  now : Int64,
) -> Array[String] {
  let tracker = self.metrics
    .get(snapshot.name())
    .unwrap_or(RunningMoments::new())
  tracker.update(snapshot.value())
  self.metrics[snapshot.name()] = tracker
  self.snapshots += 1
  let triggered : Array[String] = []
  for rule in self.rules {
    if rule.evaluate(snapshot, now) {
      triggered.push(rule.name())
      self.alerts += 1
    }
  }
  triggered
}

///|
pub fn ModelMonitor::mean(self : ModelMonitor, metric : String) -> Double {
  self.metrics.get(metric).map(value => value.mean()).unwrap_or(0.0)
}

///|
pub fn ModelMonitor::variance(self : ModelMonitor, metric : String) -> Double {
  self.metrics.get(metric).map(value => value.variance()).unwrap_or(0.0)
}

///|
pub fn ModelMonitor::snapshot_count(self : ModelMonitor) -> Int {
  self.snapshots
}

///|
pub fn ModelMonitor::alert_count(self : ModelMonitor) -> Int {
  self.alerts
}

///|
pub fn ModelMonitor::reset(self : ModelMonitor) -> Unit {
  self.metrics.clear()
  self.snapshots = 0
  self.alerts = 0
  for rule in self.rules {
    rule.reset()
  }
}

///|
pub struct ThroughputTracker {
  capacity : Int
  events : SequenceWindow
  mut accepted : Int
  mut dropped : Int
}

///|
pub fn ThroughputTracker::new(capacity? : Int = 128) -> ThroughputTracker {
  {
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
    events: SequenceWindow::new(capacity),
    accepted: 0,
    dropped: 0,
  }
}

///|
pub fn ThroughputTracker::observe(
  self : ThroughputTracker,
  timestamp : Int64,
) -> Bool {
  if timestamp < 0 {
    self.dropped += 1
    false
  } else {
    self.events.push(timestamp.to_double())
    self.accepted += 1
    true
  }
}

///|
pub fn ThroughputTracker::rate(
  self : ThroughputTracker,
  duration : Double,
) -> Double {
  if duration <= 0.0 {
    0.0
  } else {
    self.events.size().to_double() / duration
  }
}

///|
pub fn ThroughputTracker::accepted(self : ThroughputTracker) -> Int {
  self.accepted
}

///|
pub fn ThroughputTracker::dropped(self : ThroughputTracker) -> Int {
  self.dropped
}
