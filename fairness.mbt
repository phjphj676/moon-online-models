///|
/// Group-level diagnostics for online classification services.
pub struct GroupMetric {
  group : String
  mut count : Int
  mut positives : Int
  mut true_positives : Int
  mut false_positives : Int
  mut false_negatives : Int
}

///|
pub fn GroupMetric::new(group : String) -> GroupMetric {
  {
    group,
    count: 0,
    positives: 0,
    true_positives: 0,
    false_positives: 0,
    false_negatives: 0,
  }
}

///|
pub fn GroupMetric::observe(
  self : GroupMetric,
  prediction : Double,
  label : Double,
  threshold? : Double = 0.5,
) -> Unit {
  let predicted_positive = prediction >= threshold
  let actual_positive = label >= 0.5
  self.count += 1
  if actual_positive {
    self.positives += 1
  }
  if predicted_positive && actual_positive {
    self.true_positives += 1
  }
  if predicted_positive && !actual_positive {
    self.false_positives += 1
  }
  if !predicted_positive && actual_positive {
    self.false_negatives += 1
  }
}

///|
pub fn GroupMetric::group(self : GroupMetric) -> String {
  self.group
}

///|
pub fn GroupMetric::count(self : GroupMetric) -> Int {
  self.count
}

///|
pub fn GroupMetric::positive_rate(self : GroupMetric) -> Double {
  if self.count == 0 {
    0.0
  } else {
    (self.true_positives + self.false_positives).to_double() /
    self.count.to_double()
  }
}

///|
pub fn GroupMetric::recall(self : GroupMetric) -> Double {
  if self.positives == 0 {
    0.0
  } else {
    self.true_positives.to_double() / self.positives.to_double()
  }
}

///|
pub fn GroupMetric::precision(self : GroupMetric) -> Double {
  let predicted = self.true_positives + self.false_positives
  if predicted == 0 {
    0.0
  } else {
    self.true_positives.to_double() / predicted.to_double()
  }
}

///|
pub fn GroupMetric::false_positive_rate(self : GroupMetric) -> Double {
  let negatives = self.count - self.positives
  if negatives <= 0 {
    0.0
  } else {
    self.false_positives.to_double() / negatives.to_double()
  }
}

///|
pub fn GroupMetric::reset(self : GroupMetric) -> Unit {
  self.count = 0
  self.positives = 0
  self.true_positives = 0
  self.false_positives = 0
  self.false_negatives = 0
}

///|
pub struct FairnessMonitor {
  groups : Map[String, GroupMetric]
  mut observations : Int
}

///|
pub fn FairnessMonitor::new() -> FairnessMonitor {
  { groups: {}, observations: 0 }
}

///|
pub fn FairnessMonitor::observe(
  self : FairnessMonitor,
  group : String,
  prediction : Double,
  label : Double,
  threshold? : Double = 0.5,
) -> Unit {
  let metric = self.groups.get(group).unwrap_or(GroupMetric::new(group))
  metric.observe(prediction, label, threshold~)
  self.groups[group] = metric
  self.observations += 1
}

///|
pub fn FairnessMonitor::group(
  self : FairnessMonitor,
  group : String,
) -> GroupMetric? {
  self.groups.get(group)
}

///|
pub fn FairnessMonitor::groups(self : FairnessMonitor) -> Array[String] {
  self.groups.keys().to_array()
}

///|
pub fn FairnessMonitor::observations(self : FairnessMonitor) -> Int {
  self.observations
}

///|
pub fn FairnessMonitor::max_positive_rate_gap(self : FairnessMonitor) -> Double {
  let names = self.groups.keys().to_array()
  if names.length() < 2 {
    0.0
  } else {
    let mut minimum = 1.0
    let mut maximum = 0.0
    for name in names {
      let rate = self.groups
        .get(name)
        .map(metric => metric.positive_rate())
        .unwrap_or(0.0)
      if rate < minimum {
        minimum = rate
      }
      if rate > maximum {
        maximum = rate
      }
    }
    maximum - minimum
  }
}

///|
pub fn FairnessMonitor::max_recall_gap(self : FairnessMonitor) -> Double {
  let names = self.groups.keys().to_array()
  if names.length() < 2 {
    0.0
  } else {
    let mut minimum = 1.0
    let mut maximum = 0.0
    for name in names {
      let recall = self.groups
        .get(name)
        .map(metric => metric.recall())
        .unwrap_or(0.0)
      if recall < minimum {
        minimum = recall
      }
      if recall > maximum {
        maximum = recall
      }
    }
    maximum - minimum
  }
}

///|
pub fn FairnessMonitor::reset(self : FairnessMonitor) -> Unit {
  self.groups.clear()
  self.observations = 0
}
