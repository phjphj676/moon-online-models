///|
/// An owned dense vector with explicit dimension-safe operations.
///
/// `DenseVector` is useful at API boundaries where callers want a named value
/// rather than a raw array. Model hot paths still accept `Array[Double]` to
/// keep interop with MoonBit data processing code allocation-free.
pub struct DenseVector {
  values : Array[Double]
} derive(ToJson, FromJson, Debug)

///|
pub fn DenseVector::new(size : Int, fill? : Double = 0.0) -> DenseVector {
  { values: Array::make(if size < 0 { 0 } else { size }, fill) }
}

///|
pub fn DenseVector::from_array(values : Array[Double]) -> DenseVector {
  { values: copy_vector(values) }
}

///|
pub fn DenseVector::from_view(values : ArrayView[Double]) -> DenseVector {
  { values: Array::makei(values.length(), i => values[i]) }
}

///|
pub fn DenseVector::zeros(size : Int) -> DenseVector {
  DenseVector::new(size)
}

///|
pub fn DenseVector::ones(size : Int) -> DenseVector {
  DenseVector::new(size, fill=1.0)
}

///|
pub fn DenseVector::values(self : DenseVector) -> Array[Double] {
  copy_vector(self.values)
}

///|
pub fn DenseVector::dimension(self : DenseVector) -> Int {
  self.values.length()
}

///|
pub fn DenseVector::get(self : DenseVector, index : Int) -> Double? {
  self.values.get(index)
}

///|
pub fn DenseVector::set(
  self : DenseVector,
  index : Int,
  value : Double,
) -> Bool {
  if index < 0 || index >= self.values.length() {
    false
  } else {
    self.values[index] = value
    true
  }
}

///|
pub fn DenseVector::fill(self : DenseVector, value : Double) -> Unit {
  self.values.fill(value)
}

///|
pub fn DenseVector::add(self : DenseVector, other : DenseVector) -> DenseVector {
  DenseVector::from_array(add_values(self.values, other.values))
}

///|
pub fn DenseVector::subtract(
  self : DenseVector,
  other : DenseVector,
) -> DenseVector {
  DenseVector::from_array(subtract_values(self.values, other.values))
}

///|
pub fn DenseVector::scale(self : DenseVector, factor : Double) -> DenseVector {
  DenseVector::from_array(scale_values(self.values, factor))
}

///|
pub fn DenseVector::hadamard(
  self : DenseVector,
  other : DenseVector,
) -> DenseVector {
  DenseVector::from_array(hadamard_product(self.values, other.values))
}

///|
pub fn DenseVector::dot(self : DenseVector, other : DenseVector) -> Double {
  dot_product(self.values, other.values)
}

///|
pub fn DenseVector::dot_checked(
  self : DenseVector,
  other : DenseVector,
) -> Double? {
  dot_product_checked(self.values, other.values)
}

///|
pub fn DenseVector::l1_norm(self : DenseVector) -> Double {
  l1_norm(self.values)
}

///|
pub fn DenseVector::l2_norm(self : DenseVector) -> Double {
  squared_norm(self.values).sqrt()
}

///|
pub fn DenseVector::max_abs(self : DenseVector) -> Double {
  max_abs(self.values)
}

///|
pub fn DenseVector::normalize(self : DenseVector) -> DenseVector {
  DenseVector::from_array(normalize_l2(self.values))
}

///|
pub fn DenseVector::cosine(self : DenseVector, other : DenseVector) -> Double {
  cosine_similarity(self.values, other.values)
}

///|
pub fn DenseVector::sum(self : DenseVector) -> Double {
  sum_values(self.values)
}

///|
pub fn DenseVector::mean(self : DenseVector) -> Double {
  mean_values(self.values)
}

///|
pub fn DenseVector::variance(self : DenseVector) -> Double {
  variance_values(self.values)
}

///|
pub fn DenseVector::distance(self : DenseVector, other : DenseVector) -> Double {
  let size = if self.values.length() < other.values.length() {
    self.values.length()
  } else {
    other.values.length()
  }
  let mut total = 0.0
  for i in 0..<size {
    let error = self.values[i] - other.values[i]
    total += error * error
  }
  total.sqrt()
}

///|
pub fn DenseVector::map(
  self : DenseVector,
  transform : (Double) -> Double,
) -> DenseVector {
  DenseVector::from_array(
    Array::makei(self.values.length(), i => transform(self.values[i])),
  )
}

///|
pub fn DenseVector::zip_map(
  self : DenseVector,
  other : DenseVector,
  transform : (Double, Double) -> Double,
) -> DenseVector {
  let size = if self.values.length() < other.values.length() {
    self.values.length()
  } else {
    other.values.length()
  }
  DenseVector::from_array(
    Array::makei(size, i => transform(self.values[i], other.values[i])),
  )
}

///|
pub fn DenseVector::axpy(
  self : DenseVector,
  factor : Double,
  other : DenseVector,
) -> Unit {
  add_scaled_in_place(self.values, other.values, factor)
}

///|
pub fn DenseVector::clamp(
  self : DenseVector,
  lower : Double,
  upper : Double,
) -> DenseVector {
  self.map(value => clamp(value, lower, upper))
}

///|
pub fn DenseVector::softmax(self : DenseVector) -> DenseVector {
  DenseVector::from_array(probabilities_from_logits(self.values))
}

///|
pub fn DenseVector::argmax(self : DenseVector) -> Int? {
  argmax(self.values)
}

///|
pub fn DenseVector::top_index(self : DenseVector, rank : Int) -> Int? {
  top_index(self.values, rank)
}

///|
pub fn DenseVector::append(self : DenseVector, value : Double) -> Unit {
  self.values.push(value)
}

///|
pub fn DenseVector::is_empty(self : DenseVector) -> Bool {
  self.values.is_empty()
}

///|
pub fn DenseVector::is_zero(
  self : DenseVector,
  tolerance? : Double = 1.0e-12,
) -> Bool {
  max_abs(self.values) <= tolerance
}

///|
pub fn DenseVector::slice(
  self : DenseVector,
  start : Int,
  end : Int,
) -> DenseVector {
  let from = if start < 0 {
    0
  } else if start > self.values.length() {
    self.values.length()
  } else {
    start
  }
  let to = if end < from {
    from
  } else if end > self.values.length() {
    self.values.length()
  } else {
    end
  }
  DenseVector::from_array(Array::makei(to - from, i => self.values[from + i]))
}

///|
pub fn DenseVector::concat(
  self : DenseVector,
  other : DenseVector,
) -> DenseVector {
  let result = copy_vector(self.values)
  result.append(other.values[:])
  DenseVector::from_array(result)
}

///|
pub fn DenseVector::distance_squared(
  self : DenseVector,
  other : DenseVector,
) -> Double {
  let size = if self.values.length() < other.values.length() {
    self.values.length()
  } else {
    other.values.length()
  }
  let mut total = 0.0
  for i in 0..<size {
    let error = self.values[i] - other.values[i]
    total += error * error
  }
  total
}

///|
pub fn DenseVector::validate(
  self : DenseVector,
  expected : Int,
) -> ValidationReport {
  validate_vector(self.values, expected)
}

///|
pub struct Matrix {
  rows : Int
  cols : Int
  values : Array[Double]
} derive(ToJson, FromJson, Debug)

///|
pub fn Matrix::new(rows : Int, cols : Int, fill? : Double = 0.0) -> Matrix {
  let safe_rows = if rows < 0 { 0 } else { rows }
  let safe_cols = if cols < 0 { 0 } else { cols }
  {
    rows: safe_rows,
    cols: safe_cols,
    values: Array::make(safe_rows * safe_cols, fill),
  }
}

///|
pub fn Matrix::identity(size : Int) -> Matrix {
  let matrix = Matrix::new(size, size)
  for i in 0..<matrix.rows {
    matrix.values[i * matrix.cols + i] = 1.0
  }
  matrix
}

///|
pub fn Matrix::from_rows(rows : Array[Array[Double]]) -> Matrix {
  if rows.is_empty() {
    Matrix::new(0, 0)
  } else {
    let cols = rows[0].length()
    let matrix = Matrix::new(rows.length(), cols)
    for r in 0..<rows.length() {
      for c in 0..<cols {
        matrix.values[r * cols + c] = rows[r].get(c).unwrap_or(0.0)
      }
    }
    matrix
  }
}

///|
pub fn Matrix::rows(self : Matrix) -> Int {
  self.rows
}

///|
pub fn Matrix::cols(self : Matrix) -> Int {
  self.cols
}

///|
pub fn Matrix::get(self : Matrix, row : Int, col : Int) -> Double? {
  if row < 0 || row >= self.rows || col < 0 || col >= self.cols {
    None
  } else {
    Some(self.values[row * self.cols + col])
  }
}

///|
pub fn Matrix::set(self : Matrix, row : Int, col : Int, value : Double) -> Bool {
  if row < 0 || row >= self.rows || col < 0 || col >= self.cols {
    false
  } else {
    self.values[row * self.cols + col] = value
    true
  }
}

///|
pub fn Matrix::row(self : Matrix, index : Int) -> DenseVector? {
  if index < 0 || index >= self.rows {
    None
  } else {
    Some(
      DenseVector::from_array(
        Array::makei(self.cols, i => self.values[index * self.cols + i]),
      ),
    )
  }
}

///|
pub fn Matrix::column(self : Matrix, index : Int) -> DenseVector? {
  if index < 0 || index >= self.cols {
    None
  } else {
    Some(
      DenseVector::from_array(
        Array::makei(self.rows, i => self.values[i * self.cols + index]),
      ),
    )
  }
}

///|
pub fn Matrix::to_rows(self : Matrix) -> Array[Array[Double]] {
  Array::makei(self.rows, r => {
    Array::makei(self.cols, c => self.values[r * self.cols + c])
  })
}

///|
pub fn Matrix::transpose(self : Matrix) -> Matrix {
  let result = Matrix::new(self.cols, self.rows)
  for r in 0..<self.rows {
    for c in 0..<self.cols {
      result.values[c * self.rows + r] = self.values[r * self.cols + c]
    }
  }
  result
}

///|
pub fn Matrix::multiply_vector(
  self : Matrix,
  vector : Array[Double],
) -> Array[Double] {
  Array::makei(self.rows, r => {
    let mut total = 0.0
    for c in 0..<self.cols {
      total += self.values[r * self.cols + c] * vector.get(c).unwrap_or(0.0)
    }
    total
  })
}

///|
pub fn Matrix::multiply(self : Matrix, other : Matrix) -> Matrix {
  let result = Matrix::new(self.rows, other.cols)
  let shared = if self.cols < other.rows { self.cols } else { other.rows }
  for r in 0..<self.rows {
    for c in 0..<other.cols {
      let mut total = 0.0
      for k in 0..<shared {
        total += self.values[r * self.cols + k] *
          other.values[k * other.cols + c]
      }
      result.values[r * other.cols + c] = total
    }
  }
  result
}

///|
pub fn Matrix::add(self : Matrix, other : Matrix) -> Matrix {
  let result = Matrix::new(self.rows, self.cols)
  let size = if self.values.length() < other.values.length() {
    self.values.length()
  } else {
    other.values.length()
  }
  for i in 0..<size {
    result.values[i] = self.values[i] + other.values[i]
  }
  result
}

///|
pub fn Matrix::scale(self : Matrix, factor : Double) -> Matrix {
  {
    rows: self.rows,
    cols: self.cols,
    values: scale_values(self.values, factor),
  }
}

///|
pub fn Matrix::trace(self : Matrix) -> Double {
  let size = if self.rows < self.cols { self.rows } else { self.cols }
  let mut total = 0.0
  for i in 0..<size {
    total += self.values[i * self.cols + i]
  }
  total
}

///|
pub fn Matrix::frobenius_norm(self : Matrix) -> Double {
  squared_norm(self.values).sqrt()
}

///|
pub fn Matrix::is_symmetric(
  self : Matrix,
  tolerance? : Double = 1.0e-9,
) -> Bool {
  if self.rows != self.cols {
    false
  } else {
    let mut result = true
    for r in 0..<self.rows {
      for c in (r + 1)..<self.cols {
        if (self.values[r * self.cols + c] - self.values[c * self.cols + r]).abs() >
          tolerance {
          result = false
        }
      }
    }
    result
  }
}
