///|
/// Streaming k-means with bounded state and optional exponential adaptation.
pub struct OnlineKMeans {
  centroids : Array[Array[Double]]
  counts : Array[Double]
  learning_rate : Double
  mut assignments : Int
}

///|
pub fn OnlineKMeans::new(
  clusters : Int,
  dimension : Int,
  learning_rate? : Double = 0.05,
) -> OnlineKMeans {
  let cluster_count = if clusters < 0 { 0 } else { clusters }
  let size = if dimension < 0 { 0 } else { dimension }
  {
    centroids: Array::makei(cluster_count, class_index => {
      Array::makei(size, feature_index => {
        if class_index == feature_index % (cluster_count + 1) {
          1.0
        } else {
          0.0
        }
      })
    }),
    counts: Array::make(cluster_count, 0.0),
    learning_rate: clamp(learning_rate, 1.0e-6, 1.0),
    assignments: 0,
  }
}

///|
pub fn OnlineKMeans::clusters(self : OnlineKMeans) -> Int {
  self.centroids.length()
}

///|
pub fn OnlineKMeans::dimension(self : OnlineKMeans) -> Int {
  if self.centroids.is_empty() {
    0
  } else {
    self.centroids[0].length()
  }
}

///|
pub fn OnlineKMeans::centroids(self : OnlineKMeans) -> Array[Array[Double]] {
  self.centroids.map(row => copy_vector(row))
}

///|
pub fn OnlineKMeans::nearest(
  self : OnlineKMeans,
  features : Array[Double],
) -> Int? {
  if self.centroids.is_empty() {
    None
  } else {
    let mut best = 0
    let mut best_distance = squared_distance_for_cluster(
      features,
      self.centroids[0],
    )
    for cluster in 1..<self.centroids.length() {
      let distance = squared_distance_for_cluster(
        features,
        self.centroids[cluster],
      )
      if distance < best_distance {
        best = cluster
        best_distance = distance
      }
    }
    Some(best)
  }
}

///|
fn squared_distance_for_cluster(
  left : Array[Double],
  right : Array[Double],
) -> Double {
  let size = if left.length() < right.length() {
    left.length()
  } else {
    right.length()
  }
  let mut total = 0.0
  for i in 0..<size {
    let error = left[i] - right[i]
    total += error * error
  }
  total
}

///|
pub fn OnlineKMeans::update(
  self : OnlineKMeans,
  features : Array[Double],
) -> Int? {
  match self.nearest(features) {
    None => None
    Some(cluster) => {
      let centroid = self.centroids[cluster]
      let rate = self.learning_rate / (1.0 + 0.01 * self.counts[cluster])
      let limit = if features.length() < centroid.length() {
        features.length()
      } else {
        centroid.length()
      }
      for i in 0..<limit {
        centroid[i] += rate * (features[i] - centroid[i])
      }
      self.counts[cluster] += 1.0
      self.assignments += 1
      Some(cluster)
    }
  }
}

///|
pub fn OnlineKMeans::distance_to_nearest(
  self : OnlineKMeans,
  features : Array[Double],
) -> Double {
  match self.nearest(features) {
    None => 0.0
    Some(cluster) =>
      squared_distance_for_cluster(features, self.centroids[cluster]).sqrt()
  }
}

///|
pub fn OnlineKMeans::cluster_counts(self : OnlineKMeans) -> Array[Double] {
  copy_vector(self.counts)
}

///|
pub fn OnlineKMeans::assignments(self : OnlineKMeans) -> Int {
  self.assignments
}

///|
pub fn OnlineKMeans::reset(self : OnlineKMeans) -> Unit {
  self.counts.fill(0.0)
  self.assignments = 0
}

///|
pub struct OnlineMedoid {
  center : Array[Double]
  mut count : Double
  learning_rate : Double
  mut cost : Double
}

///|
pub fn OnlineMedoid::new(
  dimension : Int,
  learning_rate? : Double = 0.05,
) -> OnlineMedoid {
  {
    center: Array::make(if dimension < 0 { 0 } else { dimension }, 0.0),
    count: 0.0,
    learning_rate: clamp(learning_rate, 1.0e-6, 1.0),
    cost: 0.0,
  }
}

///|
pub fn OnlineMedoid::update(
  self : OnlineMedoid,
  features : Array[Double],
) -> Double {
  let distance = squared_distance_for_cluster(features, self.center).sqrt()
  self.count += 1.0
  let rate = self.learning_rate / (1.0 + self.count * 0.01)
  let limit = if features.length() < self.center.length() {
    features.length()
  } else {
    self.center.length()
  }
  for i in 0..<limit {
    self.center[i] += rate * (features[i] - self.center[i])
  }
  self.cost += distance
  distance
}

///|
pub fn OnlineMedoid::center(self : OnlineMedoid) -> Array[Double] {
  copy_vector(self.center)
}

///|
pub fn OnlineMedoid::count(self : OnlineMedoid) -> Double {
  self.count
}

///|
pub fn OnlineMedoid::mean_cost(self : OnlineMedoid) -> Double {
  if self.count <= 0.0 {
    0.0
  } else {
    self.cost / self.count
  }
}

///|
pub fn OnlineMedoid::reset(self : OnlineMedoid) -> Unit {
  self.center.fill(0.0)
  self.count = 0.0
  self.cost = 0.0
}
