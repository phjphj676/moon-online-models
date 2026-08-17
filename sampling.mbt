///|
/// Small deterministic pseudo-random generator for reproducible benchmarks.
pub struct DeterministicRng {
  mut state : UInt64
}

///|
pub fn DeterministicRng::new(seed : UInt64) -> DeterministicRng {
  { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed } }
}

///|
pub fn DeterministicRng::next_u64(self : DeterministicRng) -> UInt64 {
  self.state = self.state * 6364136223846793005 + 1442695040888963407
  self.state
}

///|
pub fn DeterministicRng::next_double(self : DeterministicRng) -> Double {
  let value = self.next_u64() >> 11
  value.to_double() / 9007199254740992.0
}

///|
pub fn DeterministicRng::next_int(self : DeterministicRng, limit : Int) -> Int {
  if limit <= 0 {
    0
  } else {
    (self.next_double() * limit.to_double()).to_int()
  }
}

///|
pub fn DeterministicRng::uniform(
  self : DeterministicRng,
  lower : Double,
  upper : Double,
) -> Double {
  lower + (upper - lower) * self.next_double()
}

///|
pub fn DeterministicRng::normal(self : DeterministicRng) -> Double {
  let u1 = if self.next_double() < 1.0e-15 {
    1.0e-15
  } else {
    self.next_double()
  }
  let u2 = self.next_double()
  (-2.0 * @math.ln(u1)).sqrt() * @math.cos(2.0 * @math.PI * u2)
}

///|
pub fn DeterministicRng::shuffle(
  self : DeterministicRng,
  values : Array[Int],
) -> Unit {
  if values.length() > 1 {
    for i in 0..<values.length() {
      values.swap(i, self.next_int(values.length()))
    }
  }
}

///|
pub struct ReservoirSampler {
  capacity : Int
  values : Array[Double]
  mut seen : Int
  rng : DeterministicRng
}

///|
pub fn ReservoirSampler::new(
  capacity : Int,
  seed? : UInt64 = 1,
) -> ReservoirSampler {
  {
    capacity: if capacity < 0 {
      0
    } else {
      capacity
    },
    values: [],
    seen: 0,
    rng: DeterministicRng::new(seed),
  }
}

///|
pub fn ReservoirSampler::observe(
  self : ReservoirSampler,
  value : Double,
) -> Bool {
  self.seen += 1
  if self.capacity == 0 {
    false
  } else if self.values.length() < self.capacity {
    self.values.push(value)
    true
  } else {
    let index = self.rng.next_int(self.seen)
    if index < self.capacity {
      self.values[index] = value
      true
    } else {
      false
    }
  }
}

///|
pub fn ReservoirSampler::sample(self : ReservoirSampler) -> Array[Double] {
  copy_vector(self.values)
}

///|
pub fn ReservoirSampler::capacity(self : ReservoirSampler) -> Int {
  self.capacity
}

///|
pub fn ReservoirSampler::seen(self : ReservoirSampler) -> Int {
  self.seen
}

///|
pub fn ReservoirSampler::reset(self : ReservoirSampler) -> Unit {
  self.values.clear()
  self.seen = 0
}

///|
pub struct StratifiedSampler {
  capacity_per_class : Int
  samples : Map[Int, Array[Array[Double]]]
  mut seen : Int
  rng : DeterministicRng
}

///|
pub fn StratifiedSampler::new(
  capacity_per_class : Int,
  seed? : UInt64 = 1,
) -> StratifiedSampler {
  {
    capacity_per_class: if capacity_per_class < 0 {
      0
    } else {
      capacity_per_class
    },
    samples: {},
    seen: 0,
    rng: DeterministicRng::new(seed),
  }
}

///|
pub fn StratifiedSampler::observe(
  self : StratifiedSampler,
  label : Int,
  features : Array[Double],
) -> Bool {
  self.seen += 1
  let bucket = self.samples.get_or_init(label, () => [])
  if self.capacity_per_class == 0 {
    false
  } else if bucket.length() < self.capacity_per_class {
    bucket.push(copy_vector(features))
    true
  } else {
    let index = self.rng.next_int(self.seen)
    if index < self.capacity_per_class {
      bucket[index] = copy_vector(features)
      true
    } else {
      false
    }
  }
}

///|
pub fn StratifiedSampler::class_count(
  self : StratifiedSampler,
  label : Int,
) -> Int {
  self.samples.get(label).map(value => value.length()).unwrap_or(0)
}

///|
pub fn StratifiedSampler::classes(self : StratifiedSampler) -> Array[Int] {
  self.samples.keys().to_array()
}

///|
pub fn StratifiedSampler::samples(
  self : StratifiedSampler,
  label : Int,
) -> Array[Array[Double]] {
  self.samples
  .get(label)
  .map(value => value.map(row => copy_vector(row)))
  .unwrap_or([])
}

///|
pub fn StratifiedSampler::seen(self : StratifiedSampler) -> Int {
  self.seen
}

///|
pub fn StratifiedSampler::reset(self : StratifiedSampler) -> Unit {
  self.samples.clear()
  self.seen = 0
}

///|
pub struct BootstrapCounter {
  counts : Array[Int]
  rng : DeterministicRng
  mut rounds : Int
}

///|
pub fn BootstrapCounter::new(
  size : Int,
  seed? : UInt64 = 1,
) -> BootstrapCounter {
  {
    counts: Array::make(if size < 0 { 0 } else { size }, 0),
    rng: DeterministicRng::new(seed),
    rounds: 0,
  }
}

///|
pub fn BootstrapCounter::draw(self : BootstrapCounter) -> Array[Int] {
  self.counts.fill(0)
  self.rounds += 1
  if !self.counts.is_empty() {
    for _ in 0..<self.counts.length() {
      self.counts[self.rng.next_int(self.counts.length())] += 1
    }
  }
  self.counts.map(value => value)
}

///|
pub fn BootstrapCounter::counts(self : BootstrapCounter) -> Array[Int] {
  self.counts.map(value => value)
}

///|
pub fn BootstrapCounter::rounds(self : BootstrapCounter) -> Int {
  self.rounds
}

///|
pub fn BootstrapCounter::coverage(self : BootstrapCounter) -> Double {
  if self.counts.is_empty() {
    0.0
  } else {
    self.counts.count_if(value => value > 0).to_double() /
    self.counts.length().to_double()
  }
}

///|
pub fn BootstrapCounter::reset(self : BootstrapCounter) -> Unit {
  self.counts.fill(0)
  self.rounds = 0
}

///|
pub struct BernoulliSampler {
  probability : Double
  rng : DeterministicRng
  mut accepted : Int
  mut seen : Int
}

///|
pub fn BernoulliSampler::new(
  probability : Double,
  seed? : UInt64 = 1,
) -> BernoulliSampler {
  {
    probability: clamp(probability, 0.0, 1.0),
    rng: DeterministicRng::new(seed),
    accepted: 0,
    seen: 0,
  }
}

///|
pub fn BernoulliSampler::accept(self : BernoulliSampler) -> Bool {
  self.seen += 1
  let accepted = self.rng.next_double() < self.probability
  if accepted {
    self.accepted += 1
  }
  accepted
}

///|
pub fn BernoulliSampler::probability(self : BernoulliSampler) -> Double {
  self.probability
}

///|
pub fn BernoulliSampler::seen(self : BernoulliSampler) -> Int {
  self.seen
}

///|
pub fn BernoulliSampler::accepted(self : BernoulliSampler) -> Int {
  self.accepted
}

///|
pub fn BernoulliSampler::rate(self : BernoulliSampler) -> Double {
  if self.seen == 0 {
    0.0
  } else {
    self.accepted.to_double() / self.seen.to_double()
  }
}

///|
pub fn BernoulliSampler::reset(self : BernoulliSampler) -> Unit {
  self.accepted = 0
  self.seen = 0
}

///|
pub fn sample_indices(size : Int, seed? : UInt64 = 1) -> Array[Int] {
  let result = Array::makei(if size < 0 { 0 } else { size }, i => i)
  DeterministicRng::new(seed).shuffle(result)
  result
}

///|
pub fn split_indices(
  size : Int,
  train_ratio? : Double = 0.8,
  seed? : UInt64 = 1,
) -> (Array[Int], Array[Int]) {
  let shuffled = sample_indices(size, seed~)
  let cut = (shuffled.length().to_double() * clamp(train_ratio, 0.0, 1.0)).to_int()
  (shuffled[:cut].to_owned(), shuffled[cut:].to_owned())
}
