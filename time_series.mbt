///|
/// Online autoregressive model with a fixed lag window.
pub struct OnlineAutoregressive {
  lags : Int
  weights : Array[Double]
  history : Array[Double]
  learning_rate : Double
  l2 : Double
  mut steps : Int
}

///|
pub fn OnlineAutoregressive::new(
  lags : Int,
  learning_rate? : Double = 0.01,
  l2? : Double = 0.0,
) -> OnlineAutoregressive {
  let size = if lags < 0 { 0 } else { lags }
  {
    lags: size,
    weights: Array::make(size, 0.0),
    history: [],
    learning_rate,
    l2,
    steps: 0,
  }
}

///|
pub fn OnlineAutoregressive::lags(self : OnlineAutoregressive) -> Int {
  self.lags
}

///|
pub fn OnlineAutoregressive::weights(
  self : OnlineAutoregressive,
) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn OnlineAutoregressive::history(
  self : OnlineAutoregressive,
) -> Array[Double] {
  copy_vector(self.history)
}

///|
pub fn OnlineAutoregressive::feature_vector(
  self : OnlineAutoregressive,
) -> Array[Double] {
  Array::makei(self.lags, i => {
    let index = self.history.length() - 1 - i
    self.history.get(index).unwrap_or(0.0)
  })
}

///|
pub fn OnlineAutoregressive::predict(self : OnlineAutoregressive) -> Double {
  dot_product(self.weights, self.feature_vector())
}

///|
pub fn OnlineAutoregressive::update(
  self : OnlineAutoregressive,
  value : Double,
) -> Double {
  let prediction = self.predict()
  if self.history.length() >= self.lags && self.lags > 0 {
    let error = prediction - value
    let features = self.feature_vector()
    for i in 0..<self.weights.length() {
      self.weights[i] -= self.learning_rate *
        (error * features[i] + self.l2 * self.weights[i])
    }
  }
  self.history.push(value)
  if self.history.length() > self.lags && self.lags > 0 {
    let _ = self.history.remove(0)
  }
  self.steps += 1
  prediction
}

///|
pub fn OnlineAutoregressive::forecast(
  self : OnlineAutoregressive,
  horizon : Int,
) -> Array[Double] {
  let state = copy_vector(self.history)
  let predictions = Array::make(0, 0.0)
  let safe_horizon = if horizon < 0 { 0 } else { horizon }
  for _ in 0..<safe_horizon {
    let features = Array::makei(self.lags, i => {
      let index = state.length() - 1 - i
      state.get(index).unwrap_or(0.0)
    })
    let prediction = dot_product(self.weights, features)
    predictions.push(prediction)
    state.push(prediction)
    if state.length() > self.lags && self.lags > 0 {
      let _ = state.remove(0)
    }
  }
  predictions
}

///|
pub fn OnlineAutoregressive::steps(self : OnlineAutoregressive) -> Int {
  self.steps
}

///|
pub fn OnlineAutoregressive::reset(self : OnlineAutoregressive) -> Unit {
  self.weights.fill(0.0)
  self.history.clear()
  self.steps = 0
}

///|
/// Holt-Winters style level/trend/seasonal smoother.
pub struct HoltWinters {
  alpha : Double
  beta : Double
  gamma : Double
  season_length : Int
  seasonals : Array[Double]
  mut level : Double
  mut trend : Double
  mut count : Int
}

///|
pub fn HoltWinters::new(
  season_length : Int,
  alpha? : Double = 0.2,
  beta? : Double = 0.05,
  gamma? : Double = 0.1,
) -> HoltWinters {
  let size = if season_length < 1 { 1 } else { season_length }
  {
    alpha: clamp(alpha, 1.0e-6, 1.0),
    beta: clamp(beta, 1.0e-6, 1.0),
    gamma: clamp(gamma, 1.0e-6, 1.0),
    season_length: size,
    seasonals: Array::make(size, 0.0),
    level: 0.0,
    trend: 0.0,
    count: 0,
  }
}

///|
pub fn HoltWinters::update(self : HoltWinters, value : Double) -> Double {
  let season_index = self.count % self.season_length
  if self.count == 0 {
    self.level = value
    self.trend = 0.0
  } else {
    let previous_level = self.level
    let seasonal = self.seasonals[season_index]
    self.level = self.alpha * (value - seasonal) +
      (1.0 - self.alpha) * (self.level + self.trend)
    self.trend = self.beta * (self.level - previous_level) +
      (1.0 - self.beta) * self.trend
    self.seasonals[season_index] = self.gamma * (value - self.level) +
      (1.0 - self.gamma) * seasonal
  }
  self.count += 1
  self.level + self.trend + self.seasonals[season_index]
}

///|
pub fn HoltWinters::forecast(
  self : HoltWinters,
  horizon : Int,
) -> Array[Double] {
  let safe_horizon = if horizon < 0 { 0 } else { horizon }
  Array::makei(safe_horizon, i => {
    let season_index = (self.count + i) % self.season_length
    self.level + (i + 1).to_double() * self.trend + self.seasonals[season_index]
  })
}

///|
pub fn HoltWinters::level(self : HoltWinters) -> Double {
  self.level
}

///|
pub fn HoltWinters::trend(self : HoltWinters) -> Double {
  self.trend
}

///|
pub fn HoltWinters::seasonals(self : HoltWinters) -> Array[Double] {
  copy_vector(self.seasonals)
}

///|
pub fn HoltWinters::count(self : HoltWinters) -> Int {
  self.count
}

///|
pub fn HoltWinters::reset(self : HoltWinters) -> Unit {
  self.seasonals.fill(0.0)
  self.level = 0.0
  self.trend = 0.0
  self.count = 0
}

///|
/// Seasonal baseline using per-phase running means.
pub struct SeasonalMean {
  sums : Array[Double]
  counts : Array[Double]
  mut observations : Int
}

///|
pub fn SeasonalMean::new(period : Int) -> SeasonalMean {
  let size = if period < 1 { 1 } else { period }
  {
    sums: Array::make(size, 0.0),
    counts: Array::make(size, 0.0),
    observations: 0,
  }
}

///|
pub fn SeasonalMean::period(self : SeasonalMean) -> Int {
  self.sums.length()
}

///|
pub fn SeasonalMean::update(self : SeasonalMean, value : Double) -> Double {
  let index = self.observations % self.period()
  self.sums[index] += value
  self.counts[index] += 1.0
  self.observations += 1
  self.mean_at(index)
}

///|
pub fn SeasonalMean::mean_at(self : SeasonalMean, index : Int) -> Double {
  if index < 0 || index >= self.period() || self.counts[index] <= 0.0 {
    0.0
  } else {
    self.sums[index] / self.counts[index]
  }
}

///|
pub fn SeasonalMean::forecast(
  self : SeasonalMean,
  horizon : Int,
) -> Array[Double] {
  let safe_horizon = if horizon < 0 { 0 } else { horizon }
  Array::makei(safe_horizon, i => {
    self.mean_at((self.observations + i) % self.period())
  })
}

///|
pub fn SeasonalMean::observations(self : SeasonalMean) -> Int {
  self.observations
}

///|
pub fn SeasonalMean::reset(self : SeasonalMean) -> Unit {
  self.sums.fill(0.0)
  self.counts.fill(0.0)
  self.observations = 0
}

///|
pub struct ForecastTracker {
  metrics : RegressionMetrics
  mut predictions : Int
}

///|
pub fn ForecastTracker::new() -> ForecastTracker {
  { metrics: RegressionMetrics::new(), predictions: 0 }
}

///|
pub fn ForecastTracker::observe(
  self : ForecastTracker,
  prediction : Double,
  actual : Double,
) -> Unit {
  self.metrics.update(prediction, actual)
  self.predictions += 1
}

///|
pub fn ForecastTracker::mae(self : ForecastTracker) -> Double {
  self.metrics.mae()
}

///|
pub fn ForecastTracker::rmse(self : ForecastTracker) -> Double {
  self.metrics.rmse()
}

///|
pub fn ForecastTracker::r2(self : ForecastTracker) -> Double {
  self.metrics.r2()
}

///|
pub fn ForecastTracker::predictions(self : ForecastTracker) -> Int {
  self.predictions
}

///|
pub fn ForecastTracker::reset(self : ForecastTracker) -> Unit {
  self.metrics.reset()
  self.predictions = 0
}
