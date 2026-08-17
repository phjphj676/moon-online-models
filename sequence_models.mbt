///|
/// Compact sequence-learning primitives for event streams and demand signals.
pub struct SequenceWindow {
  capacity : Int
  values : Array[Double]
  mut total : Double
  mut sum_squares : Double
}

///|
pub fn SequenceWindow::new(capacity : Int) -> SequenceWindow {
  {
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
    values: [],
    total: 0.0,
    sum_squares: 0.0,
  }
}

///|
pub fn SequenceWindow::push(self : SequenceWindow, value : Double) -> Unit {
  self.values.push(value)
  self.total += value
  self.sum_squares += value * value
  if self.values.length() > self.capacity {
    let removed = self.values.remove(0)
    self.total -= removed
    self.sum_squares -= removed * removed
  }
}

///|
pub fn SequenceWindow::size(self : SequenceWindow) -> Int {
  self.values.length()
}

///|
pub fn SequenceWindow::capacity(self : SequenceWindow) -> Int {
  self.capacity
}

///|
pub fn SequenceWindow::mean(self : SequenceWindow) -> Double {
  if self.values.is_empty() {
    0.0
  } else {
    self.total / self.values.length().to_double()
  }
}

///|
pub fn SequenceWindow::variance(self : SequenceWindow) -> Double {
  if self.values.is_empty() {
    0.0
  } else {
    let mean = self.mean()
    let value = self.sum_squares / self.values.length().to_double() -
      mean * mean
    if value < 0.0 {
      0.0
    } else {
      value
    }
  }
}

///|
pub fn SequenceWindow::last(self : SequenceWindow) -> Double {
  self.values.last().unwrap_or(0.0)
}

///|
pub fn SequenceWindow::values(self : SequenceWindow) -> Array[Double] {
  copy_vector(self.values)
}

///|
pub fn SequenceWindow::clear(self : SequenceWindow) -> Unit {
  self.values.clear()
  self.total = 0.0
  self.sum_squares = 0.0
}

///|
pub struct OnlineAR {
  lags : Int
  coefficients : Array[Double]
  history : SequenceWindow
  learning_rate : Double
  mut observations : Int
  mut squared_error : Double
}

///|
pub fn OnlineAR::new(lags : Int, learning_rate? : Double = 0.02) -> OnlineAR {
  let size = if lags < 1 { 1 } else { lags }
  {
    lags: size,
    coefficients: Array::make(size, 0.0),
    history: SequenceWindow::new(size),
    learning_rate: clamp(learning_rate, 1.0e-6, 1.0),
    observations: 0,
    squared_error: 0.0,
  }
}

///|
fn OnlineAR::sequence_prediction(self : OnlineAR) -> Double {
  let history = self.history.values()
  let mut prediction = 0.0
  let count = history.length()
  for i in 0..<self.lags {
    let offset = count - 1 - i
    if offset >= 0 {
      prediction += self.coefficients[i] * history[offset]
    }
  }
  prediction
}

///|
pub fn OnlineAR::predict(self : OnlineAR) -> Double {
  self.sequence_prediction()
}

///|
pub fn OnlineAR::update(self : OnlineAR, value : Double) -> Double {
  let prediction = self.predict()
  let error = value - prediction
  let history = self.history.values()
  let count = history.length()
  for i in 0..<self.lags {
    let offset = count - 1 - i
    if offset >= 0 {
      self.coefficients[i] += self.learning_rate * error * history[offset]
    }
  }
  self.history.push(value)
  self.observations += 1
  self.squared_error += error * error
  error
}

///|
pub fn OnlineAR::coefficients(self : OnlineAR) -> Array[Double] {
  copy_vector(self.coefficients)
}

///|
pub fn OnlineAR::observations(self : OnlineAR) -> Int {
  self.observations
}

///|
pub fn OnlineAR::rmse(self : OnlineAR) -> Double {
  if self.observations == 0 {
    0.0
  } else {
    (self.squared_error / self.observations.to_double()).sqrt()
  }
}

///|
pub fn OnlineAR::reset(self : OnlineAR) -> Unit {
  self.coefficients.fill(0.0)
  self.history.clear()
  self.observations = 0
  self.squared_error = 0.0
}

///|
pub struct MarkovTransitionModel {
  states : Int
  transitions : Array[Array[Double]]
  totals : Array[Double]
  mut previous : Int?
}

///|
pub fn MarkovTransitionModel::new(
  states : Int,
  smoothing? : Double = 1.0,
) -> MarkovTransitionModel {
  let size = if states < 0 { 0 } else { states }
  let safe_smoothing = if smoothing < 0.0 { 0.0 } else { smoothing }
  {
    states: size,
    transitions: Array::makei(size, _ => Array::make(size, safe_smoothing)),
    totals: Array::make(size, safe_smoothing * size.to_double()),
    previous: None,
  }
}

///|
pub fn MarkovTransitionModel::observe(
  self : MarkovTransitionModel,
  state : Int,
) -> Bool {
  if state < 0 || state >= self.states {
    false
  } else {
    match self.previous {
      Some(previous) => {
        self.transitions[previous][state] += 1.0
        self.totals[previous] += 1.0
      }
      None => ()
    }
    self.previous = Some(state)
    true
  }
}

///|
pub fn MarkovTransitionModel::probability(
  self : MarkovTransitionModel,
  from : Int,
  to : Int,
) -> Double {
  if from < 0 ||
    from >= self.states ||
    to < 0 ||
    to >= self.states ||
    self.totals[from] <= 0.0 {
    0.0
  } else {
    self.transitions[from][to] / self.totals[from]
  }
}

///|
pub fn MarkovTransitionModel::distribution(
  self : MarkovTransitionModel,
  from : Int,
) -> Array[Double] {
  if from < 0 || from >= self.states {
    []
  } else {
    Array::makei(self.states, to => self.probability(from, to))
  }
}

///|
pub fn MarkovTransitionModel::next_state(
  self : MarkovTransitionModel,
  from : Int,
) -> Int? {
  let distribution = self.distribution(from)
  if distribution.is_empty() {
    None
  } else {
    let mut best = 0
    for i in 1..<distribution.length() {
      if distribution[i] > distribution[best] {
        best = i
      }
    }
    Some(best)
  }
}

///|
pub fn MarkovTransitionModel::states(self : MarkovTransitionModel) -> Int {
  self.states
}

///|
pub fn MarkovTransitionModel::reset(self : MarkovTransitionModel) -> Unit {
  for row in self.transitions {
    row.fill(0.0)
  }
  self.totals.fill(0.0)
  self.previous = None
}

///|
pub struct SequenceFeatureBuilder {
  lags : Int
  include_delta : Bool
  include_mean : Bool
  include_variance : Bool
  window : SequenceWindow
}

///|
pub fn SequenceFeatureBuilder::new(
  lags : Int,
  include_delta? : Bool = true,
  include_mean? : Bool = true,
  include_variance? : Bool = true,
) -> SequenceFeatureBuilder {
  let size = if lags < 1 { 1 } else { lags }
  {
    lags: size,
    include_delta,
    include_mean,
    include_variance,
    window: SequenceWindow::new(size + 1),
  }
}

///|
pub fn SequenceFeatureBuilder::transform(
  self : SequenceFeatureBuilder,
  value : Double,
) -> Array[Double] {
  let previous = self.window.values()
  let output : Array[Double] = []
  for i in 0..<self.lags {
    let offset = previous.length() - 1 - i
    output.push(if offset >= 0 { previous[offset] } else { 0.0 })
  }
  if self.include_delta {
    output.push(
      if previous.is_empty() {
        0.0
      } else {
        value - previous.last().unwrap_or(value)
      },
    )
  }
  self.window.push(value)
  if self.include_mean {
    output.push(self.window.mean())
  }
  if self.include_variance {
    output.push(self.window.variance())
  }
  output
}

///|
pub fn SequenceFeatureBuilder::feature_count(
  self : SequenceFeatureBuilder,
) -> Int {
  self.lags +
  (if self.include_delta { 1 } else { 0 }) +
  (if self.include_mean { 1 } else { 0 }) +
  (if self.include_variance { 1 } else { 0 })
}

///|
pub fn SequenceFeatureBuilder::reset(self : SequenceFeatureBuilder) -> Unit {
  self.window.clear()
}

///|
pub struct ChangePointDetector {
  short : SequenceWindow
  long : SequenceWindow
  threshold : Double
  mut changes : Int
  mut last_score : Double
}

///|
pub fn ChangePointDetector::new(
  short_window? : Int = 8,
  long_window? : Int = 32,
  threshold? : Double = 2.5,
) -> ChangePointDetector {
  {
    short: SequenceWindow::new(short_window),
    long: SequenceWindow::new(long_window),
    threshold: if threshold < 0.0 {
      0.0
    } else {
      threshold
    },
    changes: 0,
    last_score: 0.0,
  }
}

///|
pub fn ChangePointDetector::observe(
  self : ChangePointDetector,
  value : Double,
) -> Bool {
  self.short.push(value)
  self.long.push(value)
  let spread = self.long.variance().sqrt() + 1.0e-9
  self.last_score = (self.short.mean() - self.long.mean()).abs() / spread
  let changed = self.long.size() >= self.long.capacity() &&
    self.last_score >= self.threshold
  if changed {
    self.changes += 1
  }
  changed
}

///|
pub fn ChangePointDetector::score(self : ChangePointDetector) -> Double {
  self.last_score
}

///|
pub fn ChangePointDetector::changes(self : ChangePointDetector) -> Int {
  self.changes
}

///|
pub fn ChangePointDetector::reset(self : ChangePointDetector) -> Unit {
  self.short.clear()
  self.long.clear()
  self.changes = 0
  self.last_score = 0.0
}
