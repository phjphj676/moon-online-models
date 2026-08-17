///|
/// Learning-rate schedules shared by adaptive online optimizers.
pub enum LearningRateSchedule {
  Constant
  InverseScaling(power~ : Double)
  ExponentialDecay(decay~ : Double)
  CosineDecay(minimum~ : Double, period~ : Int)
} derive(ToJson, FromJson, Debug, Eq)

///|
pub fn LearningRateSchedule::rate(
  self : LearningRateSchedule,
  initial : Double,
  step : Int,
) -> Double {
  let safe_step = if step < 0 { 0 } else { step }
  match self {
    Constant => initial
    InverseScaling(power~) =>
      initial / @math.pow(1.0 + safe_step.to_double(), power)
    ExponentialDecay(decay~) =>
      initial * @math.exp(-decay * safe_step.to_double())
    CosineDecay(minimum~, period~) => {
      let safe_period = if period <= 0 { 1 } else { period }
      let phase = (safe_step % safe_period).to_double() /
        safe_period.to_double()
      minimum + 0.5 * (initial - minimum) * (1.0 + @math.cos(@math.PI * phase))
    }
  }
}

///|
pub struct GradientClipper {
  max_norm : Double
  max_value : Double
}

///|
pub fn GradientClipper::new(
  max_norm? : Double = 0.0,
  max_value? : Double = 0.0,
) -> GradientClipper {
  {
    max_norm: if max_norm < 0.0 {
      0.0
    } else {
      max_norm
    },
    max_value: if max_value < 0.0 {
      0.0
    } else {
      max_value
    },
  }
}

///|
pub fn GradientClipper::clip(
  self : GradientClipper,
  gradients : Array[Double],
) -> Array[Double] {
  let result = copy_vector(gradients)
  if self.max_value > 0.0 {
    for i in 0..<result.length() {
      result[i] = clamp(result[i], -self.max_value, self.max_value)
    }
  }
  if self.max_norm > 0.0 {
    let norm = squared_norm(result).sqrt()
    if norm > self.max_norm {
      let scale = self.max_norm / norm
      for i in 0..<result.length() {
        result[i] *= scale
      }
    }
  }
  result
}

///|
pub fn GradientClipper::max_norm(self : GradientClipper) -> Double {
  self.max_norm
}

///|
pub fn GradientClipper::max_value(self : GradientClipper) -> Double {
  self.max_value
}

///|
pub struct OptimizerStatistics {
  mut steps : Int
  mut gradient_l1 : Double
  mut gradient_l2 : Double
  mut update_l2 : Double
}

///|
pub fn OptimizerStatistics::new() -> OptimizerStatistics {
  { steps: 0, gradient_l1: 0.0, gradient_l2: 0.0, update_l2: 0.0 }
}

///|
pub fn OptimizerStatistics::record(
  self : OptimizerStatistics,
  gradient : Array[Double],
  update : Array[Double],
) -> Unit {
  self.steps += 1
  self.gradient_l1 += l1_norm(gradient)
  self.gradient_l2 += squared_norm(gradient).sqrt()
  self.update_l2 += squared_norm(update).sqrt()
}

///|
pub fn OptimizerStatistics::steps(self : OptimizerStatistics) -> Int {
  self.steps
}

///|
pub fn OptimizerStatistics::mean_gradient_l1(
  self : OptimizerStatistics,
) -> Double {
  if self.steps == 0 {
    0.0
  } else {
    self.gradient_l1 / self.steps.to_double()
  }
}

///|
pub fn OptimizerStatistics::mean_gradient_l2(
  self : OptimizerStatistics,
) -> Double {
  if self.steps == 0 {
    0.0
  } else {
    self.gradient_l2 / self.steps.to_double()
  }
}

///|
pub fn OptimizerStatistics::mean_update_l2(
  self : OptimizerStatistics,
) -> Double {
  if self.steps == 0 {
    0.0
  } else {
    self.update_l2 / self.steps.to_double()
  }
}

///|
pub struct AdagradOptimizer {
  learning_rate : Double
  epsilon : Double
  schedule : LearningRateSchedule
  accumulator : Array[Double]
  clipper : GradientClipper
  mut step_count : Int
  statistics : OptimizerStatistics
}

///|
pub fn AdagradOptimizer::new(
  dimension : Int,
  learning_rate? : Double = 0.01,
  epsilon? : Double = 1.0e-8,
  schedule? : LearningRateSchedule = Constant,
  clipper? : GradientClipper = GradientClipper::new(),
) -> AdagradOptimizer {
  {
    learning_rate,
    epsilon,
    schedule,
    accumulator: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    clipper,
    step_count: 0,
    statistics: OptimizerStatistics::new(),
  }
}

///|
pub fn AdagradOptimizer::dimension(self : AdagradOptimizer) -> Int {
  self.accumulator.length()
}

///|
pub fn AdagradOptimizer::step_count(self : AdagradOptimizer) -> Int {
  self.step_count
}

///|
pub fn AdagradOptimizer::rate(self : AdagradOptimizer) -> Double {
  self.schedule.rate(self.learning_rate, self.step_count)
}

///|
pub fn AdagradOptimizer::accumulated(self : AdagradOptimizer) -> Array[Double] {
  copy_vector(self.accumulator)
}

///|
pub fn AdagradOptimizer::update(
  self : AdagradOptimizer,
  gradients : Array[Double],
) -> Array[Double] {
  let clipped = self.clipper.clip(gradients)
  let updates = Array::make(clipped.length(), 0.0)
  let rate = self.rate()
  let limit = if clipped.length() < self.accumulator.length() {
    clipped.length()
  } else {
    self.accumulator.length()
  }
  for i in 0..<limit {
    self.accumulator[i] += clipped[i] * clipped[i]
    updates[i] = rate * clipped[i] / (self.accumulator[i].sqrt() + self.epsilon)
  }
  self.step_count += 1
  self.statistics.record(clipped, updates)
  updates
}

///|
pub fn AdagradOptimizer::apply(
  self : AdagradOptimizer,
  parameters : Array[Double],
  gradients : Array[Double],
) -> Unit {
  let updates = self.update(gradients)
  let limit = if parameters.length() < updates.length() {
    parameters.length()
  } else {
    updates.length()
  }
  for i in 0..<limit {
    parameters[i] -= updates[i]
  }
}

///|
pub fn AdagradOptimizer::statistics(
  self : AdagradOptimizer,
) -> OptimizerStatistics {
  self.statistics
}

///|
pub fn AdagradOptimizer::reset(self : AdagradOptimizer) -> Unit {
  self.accumulator.fill(0.0)
  self.step_count = 0
}

///|
pub struct MomentumOptimizer {
  learning_rate : Double
  momentum : Double
  dampening : Double
  velocity : Array[Double]
  clipper : GradientClipper
  mut step_count : Int
}

///|
pub fn MomentumOptimizer::new(
  dimension : Int,
  learning_rate? : Double = 0.01,
  momentum? : Double = 0.9,
  dampening? : Double = 0.0,
  clipper? : GradientClipper = GradientClipper::new(),
) -> MomentumOptimizer {
  {
    learning_rate,
    momentum: clamp(momentum, 0.0, 0.9999),
    dampening: clamp(dampening, 0.0, 1.0),
    velocity: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    clipper,
    step_count: 0,
  }
}

///|
pub fn MomentumOptimizer::update(
  self : MomentumOptimizer,
  gradients : Array[Double],
) -> Array[Double] {
  let clipped = self.clipper.clip(gradients)
  let updates = Array::make(clipped.length(), 0.0)
  let limit = if clipped.length() < self.velocity.length() {
    clipped.length()
  } else {
    self.velocity.length()
  }
  for i in 0..<limit {
    let input = if self.step_count == 0 {
      clipped[i]
    } else {
      (1.0 - self.dampening) * clipped[i]
    }
    self.velocity[i] = self.momentum * self.velocity[i] + input
    updates[i] = self.learning_rate * self.velocity[i]
  }
  self.step_count += 1
  updates
}

///|
pub fn MomentumOptimizer::apply(
  self : MomentumOptimizer,
  parameters : Array[Double],
  gradients : Array[Double],
) -> Unit {
  let updates = self.update(gradients)
  let limit = if parameters.length() < updates.length() {
    parameters.length()
  } else {
    updates.length()
  }
  for i in 0..<limit {
    parameters[i] -= updates[i]
  }
}

///|
pub fn MomentumOptimizer::velocity(self : MomentumOptimizer) -> Array[Double] {
  copy_vector(self.velocity)
}

///|
pub fn MomentumOptimizer::reset(self : MomentumOptimizer) -> Unit {
  self.velocity.fill(0.0)
  self.step_count = 0
}

///|
pub struct RMSPropOptimizer {
  learning_rate : Double
  decay : Double
  epsilon : Double
  mean_square : Array[Double]
  clipper : GradientClipper
  mut step_count : Int
}

///|
pub fn RMSPropOptimizer::new(
  dimension : Int,
  learning_rate? : Double = 0.001,
  decay? : Double = 0.9,
  epsilon? : Double = 1.0e-8,
  clipper? : GradientClipper = GradientClipper::new(),
) -> RMSPropOptimizer {
  {
    learning_rate,
    decay: clamp(decay, 0.0, 0.9999),
    epsilon,
    mean_square: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    clipper,
    step_count: 0,
  }
}

///|
pub fn RMSPropOptimizer::update(
  self : RMSPropOptimizer,
  gradients : Array[Double],
) -> Array[Double] {
  let clipped = self.clipper.clip(gradients)
  let updates = Array::make(clipped.length(), 0.0)
  let limit = if clipped.length() < self.mean_square.length() {
    clipped.length()
  } else {
    self.mean_square.length()
  }
  for i in 0..<limit {
    self.mean_square[i] = self.decay * self.mean_square[i] +
      (1.0 - self.decay) * clipped[i] * clipped[i]
    updates[i] = self.learning_rate *
      clipped[i] /
      (self.mean_square[i].sqrt() + self.epsilon)
  }
  self.step_count += 1
  updates
}

///|
pub fn RMSPropOptimizer::apply(
  self : RMSPropOptimizer,
  parameters : Array[Double],
  gradients : Array[Double],
) -> Unit {
  let updates = self.update(gradients)
  let limit = if parameters.length() < updates.length() {
    parameters.length()
  } else {
    updates.length()
  }
  for i in 0..<limit {
    parameters[i] -= updates[i]
  }
}

///|
pub fn RMSPropOptimizer::mean_square(self : RMSPropOptimizer) -> Array[Double] {
  copy_vector(self.mean_square)
}

///|
pub fn RMSPropOptimizer::reset(self : RMSPropOptimizer) -> Unit {
  self.mean_square.fill(0.0)
  self.step_count = 0
}

///|
pub struct AdamOptimizer {
  learning_rate : Double
  beta1 : Double
  beta2 : Double
  epsilon : Double
  first_moment : Array[Double]
  second_moment : Array[Double]
  clipper : GradientClipper
  mut step_count : Int
}

///|
pub fn AdamOptimizer::new(
  dimension : Int,
  learning_rate? : Double = 0.001,
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  epsilon? : Double = 1.0e-8,
  clipper? : GradientClipper = GradientClipper::new(),
) -> AdamOptimizer {
  let size = if dimension < 0 { 0 } else { dimension }
  {
    learning_rate,
    beta1: clamp(beta1, 0.0, 0.9999),
    beta2: clamp(beta2, 0.0, 0.999999),
    epsilon,
    first_moment: Array::make(size, 0.0),
    second_moment: Array::make(size, 0.0),
    clipper,
    step_count: 0,
  }
}

///|
pub fn AdamOptimizer::update(
  self : AdamOptimizer,
  gradients : Array[Double],
) -> Array[Double] {
  let clipped = self.clipper.clip(gradients)
  let updates = Array::make(clipped.length(), 0.0)
  self.step_count += 1
  let correction1 = 1.0 - @math.pow(self.beta1, self.step_count.to_double())
  let correction2 = 1.0 - @math.pow(self.beta2, self.step_count.to_double())
  let limit = if clipped.length() < self.first_moment.length() {
    clipped.length()
  } else {
    self.first_moment.length()
  }
  for i in 0..<limit {
    self.first_moment[i] = self.beta1 * self.first_moment[i] +
      (1.0 - self.beta1) * clipped[i]
    self.second_moment[i] = self.beta2 * self.second_moment[i] +
      (1.0 - self.beta2) * clipped[i] * clipped[i]
    let mean = self.first_moment[i] / correction1
    let variance = self.second_moment[i] / correction2
    updates[i] = self.learning_rate * mean / (variance.sqrt() + self.epsilon)
  }
  updates
}

///|
pub fn AdamOptimizer::apply(
  self : AdamOptimizer,
  parameters : Array[Double],
  gradients : Array[Double],
) -> Unit {
  let updates = self.update(gradients)
  let limit = if parameters.length() < updates.length() {
    parameters.length()
  } else {
    updates.length()
  }
  for i in 0..<limit {
    parameters[i] -= updates[i]
  }
}

///|
pub fn AdamOptimizer::first_moment(self : AdamOptimizer) -> Array[Double] {
  copy_vector(self.first_moment)
}

///|
pub fn AdamOptimizer::second_moment(self : AdamOptimizer) -> Array[Double] {
  copy_vector(self.second_moment)
}

///|
pub fn AdamOptimizer::step_count(self : AdamOptimizer) -> Int {
  self.step_count
}

///|
pub fn AdamOptimizer::reset(self : AdamOptimizer) -> Unit {
  self.first_moment.fill(0.0)
  self.second_moment.fill(0.0)
  self.step_count = 0
}

///|
pub struct GradientAccumulator {
  gradients : Array[Double]
  mut count : Int
}

///|
pub fn GradientAccumulator::new(dimension : Int) -> GradientAccumulator {
  {
    gradients: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    count: 0,
  }
}

///|
pub fn GradientAccumulator::add(
  self : GradientAccumulator,
  gradients : Array[Double],
) -> Unit {
  add_scaled_in_place(self.gradients, gradients, 1.0)
  self.count += 1
}

///|
pub fn GradientAccumulator::count(self : GradientAccumulator) -> Int {
  self.count
}

///|
pub fn GradientAccumulator::sum(self : GradientAccumulator) -> Array[Double] {
  copy_vector(self.gradients)
}

///|
pub fn GradientAccumulator::mean(self : GradientAccumulator) -> Array[Double] {
  if self.count == 0 {
    copy_vector(self.gradients)
  } else {
    scale_values(self.gradients, 1.0 / self.count.to_double())
  }
}

///|
pub fn GradientAccumulator::take_mean(
  self : GradientAccumulator,
) -> Array[Double] {
  let result = self.mean()
  self.gradients.fill(0.0)
  self.count = 0
  result
}
