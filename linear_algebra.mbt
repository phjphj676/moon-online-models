///|
/// Compute a determinant with partial pivoting. The method is intended for
/// small online covariance matrices, not for large dense linear algebra.
pub fn Matrix::determinant(self : Matrix) -> Double {
  if self.rows != self.cols {
    0.0
  } else {
    let work = Matrix::{
      rows: self.rows,
      cols: self.cols,
      values: copy_vector(self.values),
    }
    let mut sign = 1.0
    let mut result = 1.0
    for column in 0..<work.cols {
      let mut pivot = column
      let mut magnitude = work.values[column * work.cols + column].abs()
      for row in (column + 1)..<work.rows {
        let candidate = work.values[row * work.cols + column].abs()
        if candidate > magnitude {
          pivot = row
          magnitude = candidate
        }
      }
      if magnitude <= 1.0e-15 {
        result = 0.0
      } else {
        if pivot != column {
          for j in 0..<work.cols {
            let temporary = work.values[column * work.cols + j]
            work.values[column * work.cols + j] = work.values[pivot * work.cols +
              j]
            work.values[pivot * work.cols + j] = temporary
          }
          sign = -sign
        }
        let diagonal = work.values[column * work.cols + column]
        result *= diagonal
        for row in (column + 1)..<work.rows {
          let factor = work.values[row * work.cols + column] / diagonal
          for j in (column + 1)..<work.cols {
            work.values[row * work.cols + j] -= factor *
              work.values[column * work.cols + j]
          }
        }
      }
    }
    sign * result
  }
}

///|
/// Solve a square linear system using Gaussian elimination with pivoting.
pub fn Matrix::solve(
  self : Matrix,
  right_hand_side : Array[Double],
) -> Array[Double]? {
  if self.rows != self.cols || right_hand_side.length() != self.rows {
    None
  } else {
    let work = Matrix::{
      rows: self.rows,
      cols: self.cols,
      values: copy_vector(self.values),
    }
    let result = copy_vector(right_hand_side)
    let mut solvable = true
    for column in 0..<work.cols {
      let mut pivot = column
      let mut magnitude = work.values[column * work.cols + column].abs()
      for row in (column + 1)..<work.rows {
        let candidate = work.values[row * work.cols + column].abs()
        if candidate > magnitude {
          pivot = row
          magnitude = candidate
        }
      }
      if magnitude <= 1.0e-15 {
        solvable = false
      } else {
        if pivot != column {
          for j in column..<work.cols {
            let temporary = work.values[column * work.cols + j]
            work.values[column * work.cols + j] = work.values[pivot * work.cols +
              j]
            work.values[pivot * work.cols + j] = temporary
          }
          let temporary_rhs = result[column]
          result[column] = result[pivot]
          result[pivot] = temporary_rhs
        }
        for row in (column + 1)..<work.rows {
          let factor = work.values[row * work.cols + column] /
            work.values[column * work.cols + column]
          for j in column..<work.cols {
            work.values[row * work.cols + j] -= factor *
              work.values[column * work.cols + j]
          }
          result[row] -= factor * result[column]
        }
      }
    }
    if !solvable {
      None
    } else {
      for offset in 0..<work.rows {
        let row = work.rows - 1 - offset
        let mut value = result[row]
        for j in (row + 1)..<work.cols {
          value -= work.values[row * work.cols + j] * result[j]
        }
        result[row] = value / work.values[row * work.cols + row]
      }
      Some(result)
    }
  }
}

///|
pub fn Matrix::inverse(self : Matrix) -> Matrix? {
  if self.rows != self.cols {
    None
  } else {
    let result = Matrix::new(self.rows, self.cols)
    let mut valid = true
    for column in 0..<self.cols {
      let basis = one_hot(column, self.rows)
      match self.solve(basis) {
        None => valid = false
        Some(solution) =>
          for row in 0..<self.rows {
            result.values[row * self.cols + column] = solution[row]
          }
      }
    }
    if valid {
      Some(result)
    } else {
      None
    }
  }
}

///|
pub fn outer_product(left : Array[Double], right : Array[Double]) -> Matrix {
  let result = Matrix::new(left.length(), right.length())
  for row in 0..<left.length() {
    for col in 0..<right.length() {
      let _ = result.set(row, col, left[row] * right[col])
    }
  }
  result
}

///|
pub fn rank_one_update(
  matrix : Matrix,
  vector : Array[Double],
  scale : Double,
) -> Unit {
  let rows = if vector.length() < matrix.rows {
    vector.length()
  } else {
    matrix.rows
  }
  let cols = if vector.length() < matrix.cols {
    vector.length()
  } else {
    matrix.cols
  }
  for row in 0..<rows {
    for col in 0..<cols {
      matrix.values[row * matrix.cols + col] += scale *
        vector[row] *
        vector[col]
    }
  }
}

///|
pub fn matrix_row_sums(matrix : Matrix) -> Array[Double] {
  Array::makei(matrix.rows, row => {
    let mut total = 0.0
    for col in 0..<matrix.cols {
      total += matrix.values[row * matrix.cols + col]
    }
    total
  })
}
