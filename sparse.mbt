///|
/// A sparse feature entry. Indices are zero-based and values equal to zero
/// are omitted from canonical sparse vectors.
pub struct SparseEntry {
  index : Int
  value : Double
} derive(ToJson, FromJson, Debug)

///|
pub fn SparseEntry::new(index : Int, value : Double) -> SparseEntry {
  { index, value }
}

///|
pub fn SparseEntry::index(self : SparseEntry) -> Int {
  self.index
}

///|
pub fn SparseEntry::value(self : SparseEntry) -> Double {
  self.value
}

///|
pub struct SparseVector {
  dimension : Int
  entries : Array[SparseEntry]
} derive(ToJson, FromJson, Debug)

///|
pub fn SparseVector::new(dimension : Int) -> SparseVector {
  { dimension: if dimension < 0 { 0 } else { dimension }, entries: [] }
}

///|
pub fn SparseVector::from_entries(
  dimension : Int,
  entries : Array[SparseEntry],
) -> SparseVector {
  let size = if dimension < 0 { 0 } else { dimension }
  let normalized = Array::make(0, SparseEntry::new(0, 0.0))
  for entry in entries {
    if entry.index >= 0 && entry.index < size && entry.value != 0.0 {
      normalized.push(entry)
    }
  }
  normalized.sort_by((left, right) => left.index - right.index)
  let compact = Array::make(0, SparseEntry::new(0, 0.0))
  for entry in normalized {
    match compact.last() {
      Some(last) =>
        if last.index == entry.index {
          compact[compact.length() - 1] = SparseEntry::new(
            last.index,
            last.value + entry.value,
          )
        } else {
          compact.push(entry)
        }
      None => compact.push(entry)
    }
  }
  compact.retain(entry => entry.value != 0.0)
  { dimension: size, entries: compact }
}

///|
pub fn SparseVector::from_dense(
  values : Array[Double],
  threshold? : Double = 0.0,
) -> SparseVector {
  let entries = Array::make(0, SparseEntry::new(0, 0.0))
  for i in 0..<values.length() {
    let value = values[i]
    let magnitude = if value < 0.0 { -value } else { value }
    if magnitude > threshold {
      entries.push(SparseEntry::new(i, value))
    }
  }
  SparseVector::from_entries(values.length(), entries)
}

///|
pub fn SparseVector::dimension(self : SparseVector) -> Int {
  self.dimension
}

///|
pub fn SparseVector::nnz(self : SparseVector) -> Int {
  self.entries.length()
}

///|
pub fn SparseVector::entries(self : SparseVector) -> Array[SparseEntry] {
  Array::makei(self.entries.length(), i => self.entries[i])
}

///|
pub fn SparseVector::get(self : SparseVector, index : Int) -> Double {
  if index < 0 || index >= self.dimension {
    0.0
  } else {
    let mut lower = 0
    let mut upper = self.entries.length()
    let mut result = 0.0
    while lower < upper {
      let middle = lower + (upper - lower) / 2
      let candidate = self.entries[middle]
      if candidate.index == index {
        result = candidate.value
        break
      } else if candidate.index < index {
        lower = middle + 1
      } else {
        upper = middle
      }
    }
    result
  }
}

///|
pub fn SparseVector::contains(self : SparseVector, index : Int) -> Bool {
  self.get(index) != 0.0
}

///|
pub fn SparseVector::to_dense(self : SparseVector) -> Array[Double] {
  let result = Array::make(self.dimension, 0.0)
  for entry in self.entries {
    result[entry.index] = entry.value
  }
  result
}

///|
pub fn SparseVector::set(
  self : SparseVector,
  index : Int,
  value : Double,
) -> SparseVector {
  let entries = self.entries
  let filtered = Array::make(0, SparseEntry::new(0, 0.0))
  let mut inserted = false
  for entry in entries {
    if entry.index == index {
      if value != 0.0 {
        filtered.push(SparseEntry::new(index, value))
      }
      inserted = true
    } else if !inserted && entry.index > index {
      if value != 0.0 {
        filtered.push(SparseEntry::new(index, value))
      }
      filtered.push(entry)
      inserted = true
    } else {
      filtered.push(entry)
    }
  }
  if !inserted && value != 0.0 {
    filtered.push(SparseEntry::new(index, value))
  }
  SparseVector::from_entries(self.dimension, filtered)
}

///|
pub fn SparseVector::add(
  self : SparseVector,
  other : SparseVector,
) -> SparseVector {
  let entries = Array::make(0, SparseEntry::new(0, 0.0))
  let left = self.entries
  let right = other.entries
  let mut i = 0
  let mut j = 0
  while i < left.length() || j < right.length() {
    if j >= right.length() ||
      (i < left.length() && left[i].index < right[j].index) {
      entries.push(left[i])
      i += 1
    } else if i >= left.length() || right[j].index < left[i].index {
      entries.push(right[j])
      j += 1
    } else {
      entries.push(
        SparseEntry::new(left[i].index, left[i].value + right[j].value),
      )
      i += 1
      j += 1
    }
  }
  SparseVector::from_entries(
    if self.dimension > other.dimension {
      self.dimension
    } else {
      other.dimension
    },
    entries,
  )
}

///|
pub fn SparseVector::subtract(
  self : SparseVector,
  other : SparseVector,
) -> SparseVector {
  let negated = other.scale(-1.0)
  self.add(negated)
}

///|
pub fn SparseVector::scale(
  self : SparseVector,
  factor : Double,
) -> SparseVector {
  SparseVector::from_entries(
    self.dimension,
    self.entries.map(entry => {
      SparseEntry::new(entry.index, entry.value * factor)
    }),
  )
}

///|
pub fn SparseVector::map(
  self : SparseVector,
  transform : (Double) -> Double,
) -> SparseVector {
  SparseVector::from_entries(
    self.dimension,
    self.entries.map(entry => {
      SparseEntry::new(entry.index, transform(entry.value))
    }),
  )
}

///|
pub fn SparseVector::dot_dense(
  self : SparseVector,
  values : Array[Double],
) -> Double {
  let mut total = 0.0
  for entry in self.entries {
    total += entry.value * values.get(entry.index).unwrap_or(0.0)
  }
  total
}

///|
pub fn SparseVector::dot_sparse(
  self : SparseVector,
  other : SparseVector,
) -> Double {
  let mut total = 0.0
  let mut i = 0
  let mut j = 0
  while i < self.entries.length() && j < other.entries.length() {
    let left = self.entries[i]
    let right = other.entries[j]
    if left.index == right.index {
      total += left.value * right.value
      i += 1
      j += 1
    } else if left.index < right.index {
      i += 1
    } else {
      j += 1
    }
  }
  total
}

///|
pub fn SparseVector::l1_norm(self : SparseVector) -> Double {
  self.entries.fold(init=0.0, (total, entry) => {
    let magnitude = if entry.value < 0.0 { -entry.value } else { entry.value }
    total + magnitude
  })
}

///|
pub fn SparseVector::l2_norm(self : SparseVector) -> Double {
  self.dot_sparse(self).sqrt()
}

///|
pub fn SparseVector::max_abs(self : SparseVector) -> Double {
  self.entries.fold(init=0.0, (result, entry) => {
    let magnitude = if entry.value < 0.0 { -entry.value } else { entry.value }
    if magnitude > result {
      magnitude
    } else {
      result
    }
  })
}

///|
pub fn SparseVector::normalize(self : SparseVector) -> SparseVector {
  let norm = self.l2_norm()
  if norm <= 1.0e-15 {
    self
  } else {
    self.scale(1.0 / norm)
  }
}

///|
pub fn SparseVector::distance_squared(
  self : SparseVector,
  other : SparseVector,
) -> Double {
  self.subtract(other).dot_sparse(self.subtract(other))
}

///|
pub fn SparseVector::cosine(
  self : SparseVector,
  other : SparseVector,
) -> Double {
  let denominator = self.l2_norm() * other.l2_norm()
  if denominator <= 1.0e-15 {
    0.0
  } else {
    self.dot_sparse(other) / denominator
  }
}

///|
pub fn SparseVector::is_empty(self : SparseVector) -> Bool {
  self.entries.is_empty()
}

///|
pub fn SparseVector::is_sorted(self : SparseVector) -> Bool {
  let mut result = true
  for i in 1..<self.entries.length() {
    if self.entries[i - 1].index >= self.entries[i].index {
      result = false
    }
  }
  result
}

///|
pub fn SparseVector::threshold(
  self : SparseVector,
  limit : Double,
) -> SparseVector {
  let entries = self.entries.filter(entry => {
    let magnitude = if entry.value < 0.0 { -entry.value } else { entry.value }
    magnitude >= limit
  })
  SparseVector::from_entries(self.dimension, entries)
}

///|
pub struct SparseAccumulator {
  dimension : Int
  values : Map[Int, Double]
}

///|
pub fn SparseAccumulator::new(dimension : Int) -> SparseAccumulator {
  { dimension: if dimension < 0 { 0 } else { dimension }, values: {} }
}

///|
pub fn SparseAccumulator::add(
  self : SparseAccumulator,
  index : Int,
  value : Double,
) -> Bool {
  if index < 0 || index >= self.dimension {
    false
  } else {
    self.values.update_or_default(index, 0.0, previous => previous + value)
    true
  }
}

///|
pub fn SparseAccumulator::clear(self : SparseAccumulator) -> Unit {
  self.values.clear()
}

///|
pub fn SparseAccumulator::to_vector(self : SparseAccumulator) -> SparseVector {
  let entries = self.values
    .to_array()
    .map(pair => SparseEntry::new(pair.0, pair.1))
  SparseVector::from_entries(self.dimension, entries)
}

///|
pub fn SparseAccumulator::get(self : SparseAccumulator, index : Int) -> Double {
  self.values.get(index).unwrap_or(0.0)
}

///|
pub fn SparseAccumulator::dimension(self : SparseAccumulator) -> Int {
  self.dimension
}

///|
pub fn hash_feature(name : String, buckets : Int) -> Int {
  if buckets <= 0 {
    0
  } else {
    let hash = name.hash()
    let positive = if hash < 0 { -hash } else { hash }
    positive % buckets
  }
}

///|
pub fn hash_feature_with_sign(name : String, buckets : Int) -> SparseEntry {
  let index = hash_feature(name, buckets)
  let hash = name.hash()
  let sign = if hash % 2 == 0 { 1.0 } else { -1.0 }
  SparseEntry::new(index, sign)
}
