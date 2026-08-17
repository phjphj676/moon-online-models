///|
/// Online matrix factorization for implicit or explicit feedback streams.
pub struct OnlineMatrixFactorization {
  users : Array[Array[Double]]
  items : Array[Array[Double]]
  learning_rate : Double
  regularization : Double
  mut updates : Int
}

///|
pub fn OnlineMatrixFactorization::new(
  users : Int,
  items : Int,
  rank : Int,
  learning_rate? : Double = 0.01,
  regularization? : Double = 0.01,
) -> OnlineMatrixFactorization {
  let user_count = if users < 0 { 0 } else { users }
  let item_count = if items < 0 { 0 } else { items }
  let factors = if rank < 0 { 0 } else { rank }
  {
    users: Array::makei(user_count, _ => Array::make(factors, 0.01)),
    items: Array::makei(item_count, _ => Array::make(factors, 0.01)),
    learning_rate,
    regularization,
    updates: 0,
  }
}

///|
pub fn OnlineMatrixFactorization::user_count(
  self : OnlineMatrixFactorization,
) -> Int {
  self.users.length()
}

///|
pub fn OnlineMatrixFactorization::item_count(
  self : OnlineMatrixFactorization,
) -> Int {
  self.items.length()
}

///|
pub fn OnlineMatrixFactorization::rank(self : OnlineMatrixFactorization) -> Int {
  if self.users.is_empty() {
    0
  } else {
    self.users[0].length()
  }
}

///|
pub fn OnlineMatrixFactorization::predict(
  self : OnlineMatrixFactorization,
  user : Int,
  item : Int,
) -> Double {
  match (self.users.get(user), self.items.get(item)) {
    (Some(left), Some(right)) => dot_product(left, right)
    _ => 0.0
  }
}

///|
pub fn OnlineMatrixFactorization::update(
  self : OnlineMatrixFactorization,
  user : Int,
  item : Int,
  rating : Double,
) -> Bool {
  match (self.users.get(user), self.items.get(item)) {
    (Some(user_vector), Some(item_vector)) => {
      let prediction = dot_product(user_vector, item_vector)
      let error = prediction - rating
      let rank = if user_vector.length() < item_vector.length() {
        user_vector.length()
      } else {
        item_vector.length()
      }
      for factor in 0..<rank {
        let user_value = user_vector[factor]
        let item_value = item_vector[factor]
        user_vector[factor] -= self.learning_rate *
          (error * item_value + self.regularization * user_value)
        item_vector[factor] -= self.learning_rate *
          (error * user_value + self.regularization * item_value)
      }
      self.updates += 1
      true
    }
    _ => false
  }
}

///|
pub fn OnlineMatrixFactorization::user_vector(
  self : OnlineMatrixFactorization,
  user : Int,
) -> Array[Double]? {
  self.users.get(user).map(vector => copy_vector(vector))
}

///|
pub fn OnlineMatrixFactorization::item_vector(
  self : OnlineMatrixFactorization,
  item : Int,
) -> Array[Double]? {
  self.items.get(item).map(vector => copy_vector(vector))
}

///|
pub fn OnlineMatrixFactorization::updates(
  self : OnlineMatrixFactorization,
) -> Int {
  self.updates
}

///|
pub fn OnlineMatrixFactorization::reset(
  self : OnlineMatrixFactorization,
) -> Unit {
  for vector in self.users {
    vector.fill(0.01)
  }
  for vector in self.items {
    vector.fill(0.01)
  }
  self.updates = 0
}

///|
pub struct FactorizationMachine {
  linear : Array[Double]
  factors : Array[Array[Double]]
  learning_rate : Double
  l2 : Double
  updates : Int
}

///|
pub fn FactorizationMachine::new(
  dimension : Int,
  rank : Int,
  learning_rate? : Double = 0.01,
  l2? : Double = 0.01,
) -> FactorizationMachine {
  let size = if dimension < 0 { 0 } else { dimension }
  let factor_count = if rank < 0 { 0 } else { rank }
  {
    linear: Array::make(size, 0.0),
    factors: Array::makei(size, _ => Array::make(factor_count, 0.01)),
    learning_rate,
    l2,
    updates: 0,
  }
}

///|
pub fn FactorizationMachine::predict(
  self : FactorizationMachine,
  features : SparseVector,
) -> Double {
  let result = Ref(features.dot_dense(self.linear))
  for factor in 0..<self.factors.get(0).map(row => row.length()).unwrap_or(0) {
    let mut sum = 0.0
    let mut square_sum = 0.0
    for entry in features.entries() {
      let value = self.factors
        .get(entry.index())
        .map(row => row[factor])
        .unwrap_or(0.0) *
        entry.value()
      sum += value
      square_sum += value * value
    }
    result.val += 0.5 * (sum * sum - square_sum)
  }
  result.val
}

///|
pub fn FactorizationMachine::updates(self : FactorizationMachine) -> Int {
  self.updates
}
