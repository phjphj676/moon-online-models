///|
/// Online principal-component analysis using Oja's rule.
pub struct OnlinePCA {
  components : Array[Array[Double]]
  mean : Array[Double]
  learning_rate : Double
  mut count : Double
  explained : Array[Double]
}

///|
pub fn OnlinePCA::new(
  components : Int,
  dimension : Int,
  learning_rate? : Double = 0.01,
) -> OnlinePCA {
  let component_count = if components < 0 { 0 } else { components }
  let size = if dimension < 0 { 0 } else { dimension }
  {
    components: Array::makei(component_count, component => {
      Array::makei(size, feature => {
        if feature == component % (size + 1) {
          1.0
        } else {
          0.0
        }
      })
    }),
    mean: Array::make(size, 0.0),
    learning_rate: clamp(learning_rate, 1.0e-6, 1.0),
    count: 0.0,
    explained: Array::make(component_count, 0.0),
  }
}

///|
pub fn OnlinePCA::dimension(self : OnlinePCA) -> Int {
  self.mean.length()
}

///|
pub fn OnlinePCA::components(self : OnlinePCA) -> Int {
  self.components.length()
}

///|
pub fn OnlinePCA::component_vectors(self : OnlinePCA) -> Array[Array[Double]] {
  self.components.map(row => copy_vector(row))
}

///|
pub fn OnlinePCA::update(self : OnlinePCA, values : Array[Double]) -> Unit {
  self.count += 1.0
  let centered = Array::makei(self.dimension(), i => {
    let value = values.get(i).unwrap_or(0.0)
    let delta = value - self.mean[i]
    self.mean[i] += delta / self.count
    value - self.mean[i]
  })
  for component_index in 0..<self.components.length() {
    let component = self.components[component_index]
    let projection = dot_product(component, centered)
    self.explained[component_index] = moving_average(
      self.explained[component_index],
      projection * projection,
      self.learning_rate,
    )
    for i in 0..<component.length() {
      component[i] += self.learning_rate *
        projection *
        (centered[i] - projection * component[i])
    }
    let norm = squared_norm(component).sqrt()
    if norm > 1.0e-15 {
      for i in 0..<component.length() {
        component[i] /= norm
      }
    }
  }
}

///|
pub fn OnlinePCA::transform(
  self : OnlinePCA,
  values : Array[Double],
) -> Array[Double] {
  let centered = Array::makei(self.dimension(), i => {
    values.get(i).unwrap_or(0.0) - self.mean[i]
  })
  self.components.map(component => dot_product(component, centered))
}

///|
pub fn OnlinePCA::reconstruct(
  self : OnlinePCA,
  transformed : Array[Double],
) -> Array[Double] {
  let result = copy_vector(self.mean)
  let limit = if transformed.length() < self.components.length() {
    transformed.length()
  } else {
    self.components.length()
  }
  for component_index in 0..<limit {
    add_scaled_in_place(
      result,
      self.components[component_index],
      transformed[component_index],
    )
  }
  result
}

///|
pub fn OnlinePCA::explained_variance(self : OnlinePCA) -> Array[Double] {
  copy_vector(self.explained)
}

///|
pub fn OnlinePCA::mean(self : OnlinePCA) -> Array[Double] {
  copy_vector(self.mean)
}

///|
pub fn OnlinePCA::count(self : OnlinePCA) -> Double {
  self.count
}

///|
pub fn OnlinePCA::reset(self : OnlinePCA) -> Unit {
  self.mean.fill(0.0)
  self.explained.fill(0.0)
  self.count = 0.0
}
