///|
/// Generic weighted counters for stream dashboards.
pub struct OnlineCounter {
  values : Map[String, Double]
  mut updates : Int
}

///|
pub fn OnlineCounter::new() -> OnlineCounter {
  { values: {}, updates: 0 }
}

///|
pub fn OnlineCounter::add(
  self : OnlineCounter,
  key : String,
  value? : Double = 1.0,
) -> Unit {
  self.values.update_or_default(key, 0.0, previous => previous + value)
  self.updates += 1
}

///|
pub fn OnlineCounter::get(self : OnlineCounter, key : String) -> Double {
  self.values.get(key).unwrap_or(0.0)
}

///|
pub fn OnlineCounter::keys(self : OnlineCounter) -> Array[String] {
  self.values.keys().to_array()
}

///|
pub fn OnlineCounter::total(self : OnlineCounter) -> Double {
  self.values.values().fold(init=0.0, (total, value) => total + value)
}

///|
pub fn OnlineCounter::updates(self : OnlineCounter) -> Int {
  self.updates
}

///|
pub fn OnlineCounter::reset(self : OnlineCounter) -> Unit {
  self.values.clear()
  self.updates = 0
}

///|
pub struct WeightedHistogram {
  lower : Double
  upper : Double
  bins : Array[Double]
  mut total : Double
}

///|
pub fn WeightedHistogram::new(
  lower? : Double = 0.0,
  upper? : Double = 1.0,
  bins? : Int = 20,
) -> WeightedHistogram {
  let count = if bins < 1 { 1 } else { bins }
  {
    lower,
    upper: if upper <= lower {
      lower + 1.0
    } else {
      upper
    },
    bins: Array::make(count, 0.0),
    total: 0.0,
  }
}

///|
pub fn WeightedHistogram::observe(
  self : WeightedHistogram,
  value : Double,
  weight? : Double = 1.0,
) -> Unit {
  let ratio = (value - self.lower) / (self.upper - self.lower)
  let index = if ratio <= 0.0 {
    0
  } else if ratio >= 1.0 {
    self.bins.length() - 1
  } else {
    (ratio * self.bins.length().to_double()).to_int()
  }
  self.bins[index] += weight
  self.total += weight
}

///|
pub fn WeightedHistogram::probabilities(
  self : WeightedHistogram,
) -> Array[Double] {
  if self.total <= 0.0 {
    Array::make(self.bins.length(), 0.0)
  } else {
    scale_values(self.bins, 1.0 / self.total)
  }
}

///|
pub fn WeightedHistogram::quantile(
  self : WeightedHistogram,
  probability : Double,
) -> Double {
  let target = clamp(probability, 0.0, 1.0) * self.total
  let mut cumulative = 0.0
  let mut index = self.bins.length() - 1
  for i in 0..<self.bins.length() {
    cumulative += self.bins[i]
    if cumulative >= target && index == self.bins.length() - 1 {
      index = i
    }
  }
  self.lower +
  (index.to_double() + 0.5) *
  (self.upper - self.lower) /
  self.bins.length().to_double()
}

///|
pub fn WeightedHistogram::total(self : WeightedHistogram) -> Double {
  self.total
}

///|
pub fn WeightedHistogram::counts(self : WeightedHistogram) -> Array[Double] {
  copy_vector(self.bins)
}

///|
pub fn WeightedHistogram::reset(self : WeightedHistogram) -> Unit {
  self.bins.fill(0.0)
  self.total = 0.0
}

///|
pub struct RateLimiter {
  limit : Int
  period : Int
  mut ticks : Int
  mut accepted : Int
}

///|
pub fn RateLimiter::new(limit : Int, period : Int) -> RateLimiter {
  {
    limit: if limit < 0 {
      0
    } else {
      limit
    },
    period: if period < 1 {
      1
    } else {
      period
    },
    ticks: 0,
    accepted: 0,
  }
}

///|
pub fn RateLimiter::allow(self : RateLimiter) -> Bool {
  self.ticks += 1
  let position = (self.ticks - 1) % self.period
  let allowed = position < self.limit
  if allowed {
    self.accepted += 1
  }
  allowed
}

///|
pub fn RateLimiter::accepted(self : RateLimiter) -> Int {
  self.accepted
}

///|
pub fn RateLimiter::reset(self : RateLimiter) -> Unit {
  self.ticks = 0
  self.accepted = 0
}
