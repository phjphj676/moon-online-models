///|
/// A scalar feature value before it is encoded into a numeric vector.
pub enum FeatureValue {
  Number(Double)
  Category(String)
  Boolean(Bool)
  Missing
} derive(ToJson, FromJson, Debug, Eq)

///|
pub fn FeatureValue::as_number(
  self : FeatureValue,
  missing_value? : Double = 0.0,
) -> Double {
  match self {
    Number(value) => value
    Boolean(value) => if value { 1.0 } else { 0.0 }
    Category(_) => missing_value
    Missing => missing_value
  }
}

///|
pub fn FeatureValue::is_missing(self : FeatureValue) -> Bool {
  match self {
    Missing => true
    _ => false
  }
}

///|
pub struct CsvOptions {
  separator : Char
  quote : Char
  escape : Char
  trim_fields : Bool
  skip_empty : Bool
}

///|
pub fn CsvOptions::new(
  separator? : Char = ',',
  quote? : Char = '"',
  escape? : Char = '"',
  trim_fields? : Bool = true,
  skip_empty? : Bool = false,
) -> CsvOptions {
  { separator, quote, escape, trim_fields, skip_empty }
}

///|
pub fn CsvOptions::separator(self : CsvOptions) -> Char {
  self.separator
}

///|
pub fn CsvOptions::parse_line(
  self : CsvOptions,
  line : String,
) -> Array[String] {
  let fields = Array::make(0, "")
  let current = Ref("")
  let quoted = Ref(false)
  for character in line {
    if character == self.quote {
      quoted.val = !quoted.val
    } else if character == self.separator && !quoted.val {
      let field = if self.trim_fields {
        current.val.trim().to_owned()
      } else {
        current.val
      }
      if !self.skip_empty || field != "" {
        fields.push(field)
      }
      current.val = ""
    } else {
      current.val = String::add(current.val, "\{character}")
    }
  }
  let field = if self.trim_fields {
    current.val.trim().to_owned()
  } else {
    current.val
  }
  if !self.skip_empty || field != "" {
    fields.push(field)
  }
  fields
}

///|
pub fn CsvOptions::parse_lines(
  self : CsvOptions,
  text : String,
) -> Array[Array[String]] {
  let rows = Array::make(0, [])
  for line in text.split("\n") {
    let value = line.to_owned()
    if value != "" || !self.skip_empty {
      rows.push(self.parse_line(value))
    }
  }
  rows
}

///|
pub fn csv_escape(
  field : String,
  options? : CsvOptions = CsvOptions::new(),
) -> String {
  let needs_quote = field.contains_char(options.separator) ||
    field.contains_char(options.quote) ||
    field.contains_char('\n')
  if !needs_quote {
    field
  } else {
    let escaped = field.replace(
      old="\{options.quote}",
      new="\{options.quote}\{options.quote}",
    )
    "\{options.quote}\{escaped}\{options.quote}"
  }
}

///|
pub fn csv_write_row(
  fields : Array[String],
  options? : CsvOptions = CsvOptions::new(),
) -> String {
  fields
  .map(field => csv_escape(field, options~))
  .join(options.separator.to_string())
}

///|
pub struct CategoricalEncoder {
  values : Map[String, Int]
  mut next_index : Int
  unknown_index : Int?
}

///|
pub fn CategoricalEncoder::new(unknown_index? : Int) -> CategoricalEncoder {
  { values: {}, next_index: 0, unknown_index }
}

///|
pub fn CategoricalEncoder::encode(
  self : CategoricalEncoder,
  value : String,
) -> Int {
  match self.values.get(value) {
    Some(index) => index
    None =>
      match self.unknown_index {
        Some(index) => index
        None => {
          let index = self.next_index
          self.values[value] = index
          self.next_index += 1
          index
        }
      }
  }
}

///|
pub fn CategoricalEncoder::contains(
  self : CategoricalEncoder,
  value : String,
) -> Bool {
  self.values.contains(value)
}

///|
pub fn CategoricalEncoder::dimension(self : CategoricalEncoder) -> Int {
  self.next_index
}

///|
pub fn CategoricalEncoder::entries(
  self : CategoricalEncoder,
) -> Array[(String, Int)] {
  self.values.to_array()
}

///|
pub fn CategoricalEncoder::reset(self : CategoricalEncoder) -> Unit {
  self.values.clear()
  self.next_index = 0
}

///|
pub struct FeatureHasher {
  buckets : Int
  signed : Bool
}

///|
pub fn FeatureHasher::new(
  buckets : Int,
  signed? : Bool = true,
) -> FeatureHasher {
  { buckets: if buckets < 0 { 0 } else { buckets }, signed }
}

///|
pub fn FeatureHasher::buckets(self : FeatureHasher) -> Int {
  self.buckets
}

///|
pub fn FeatureHasher::index(self : FeatureHasher, token : String) -> Int {
  hash_feature(token, self.buckets)
}

///|
pub fn FeatureHasher::encode(
  self : FeatureHasher,
  tokens : Array[String],
) -> SparseVector {
  let entries = Array::make(0, SparseEntry::new(0, 0.0))
  for token in tokens {
    let entry = if self.signed {
      hash_feature_with_sign(token, self.buckets)
    } else {
      SparseEntry::new(self.index(token), 1.0)
    }
    entries.push(entry)
  }
  SparseVector::from_entries(self.buckets, entries)
}

///|
pub fn FeatureHasher::encode_weighted(
  self : FeatureHasher,
  tokens : Array[(String, Double)],
) -> SparseVector {
  let entries = Array::make(0, SparseEntry::new(0, 0.0))
  for item in tokens {
    let token = item.0
    let weight = item.1
    let index = self.index(token)
    let sign = if self.signed && token.hash() % 2 != 0 { -1.0 } else { 1.0 }
    entries.push(SparseEntry::new(index, sign * weight))
  }
  SparseVector::from_entries(self.buckets, entries)
}

///|
pub fn FeatureHasher::encode_map(
  self : FeatureHasher,
  values : Map[String, Double],
) -> SparseVector {
  let entries = Array::make(0, SparseEntry::new(0, 0.0))
  values.each((token, weight) => {
    let index = self.index(token)
    let sign = if self.signed && token.hash() % 2 != 0 { -1.0 } else { 1.0 }
    entries.push(SparseEntry::new(index, sign * weight))
  })
  SparseVector::from_entries(self.buckets, entries)
}

///|
pub fn cross_feature(
  left : String,
  right : String,
  separator? : String = "=",
) -> String {
  "\{left}\{separator}\{right}"
}

///|
pub fn generate_feature_crosses(
  features : Array[String],
  order? : Int = 2,
) -> Array[String] {
  let result = Array::make(0, "")
  let safe_order = if order < 1 { 1 } else { order }
  if safe_order == 1 {
    result.append(features[:])
  } else {
    for i in 0..<features.length() {
      for j in (i + 1)..<features.length() {
        result.push(cross_feature(features[i], features[j]))
      }
    }
  }
  result
}

///|
pub struct DataBatch {
  features : Array[Array[Double]]
  labels : Array[Double]
  weights : Array[Double]
}

///|
pub fn DataBatch::new(dimension? : Int = 0) -> DataBatch {
  let _ = dimension
  { features: [], labels: [], weights: [] }
}

///|
pub fn DataBatch::from_arrays(
  features : Array[Array[Double]],
  labels : Array[Double],
) -> DataBatch {
  let weights = Array::make(labels.length(), 1.0)
  { features, labels, weights }
}

///|
pub fn DataBatch::add(
  self : DataBatch,
  features : Array[Double],
  label : Double,
  weight? : Double = 1.0,
) -> Bool {
  if !self.features.is_empty() && features.length() != self.features[0].length() {
    false
  } else {
    self.features.push(copy_vector(features))
    self.labels.push(label)
    self.weights.push(weight)
    true
  }
}

///|
pub fn DataBatch::size(self : DataBatch) -> Int {
  self.labels.length()
}

///|
pub fn DataBatch::dimension(self : DataBatch) -> Int {
  if self.features.is_empty() {
    0
  } else {
    self.features[0].length()
  }
}

///|
pub fn DataBatch::features(self : DataBatch) -> Array[Array[Double]] {
  self.features.map(row => copy_vector(row))
}

///|
pub fn DataBatch::labels(self : DataBatch) -> Array[Double] {
  copy_vector(self.labels)
}

///|
pub fn DataBatch::weights(self : DataBatch) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn DataBatch::validate(self : DataBatch) -> ValidationReport {
  if self.features.length() != self.labels.length() ||
    self.labels.length() != self.weights.length() {
    ValidationReport::error("batch arrays have different lengths")
  } else if self.features.is_empty() {
    ValidationReport::ok()
  } else {
    let dimension = self.features[0].length()
    let mut valid = true
    for row in self.features {
      if row.length() != dimension {
        valid = false
      }
    }
    if valid {
      ValidationReport::ok()
    } else {
      ValidationReport::error("batch contains inconsistent feature dimensions")
    }
  }
}

///|
pub fn DataBatch::shuffle(
  self : DataBatch,
  random_index : (Int) -> Int,
) -> Unit {
  let size = self.size()
  if size > 1 {
    for i in 0..<size {
      let j = random_index(size)
      if j >= 0 && j < size {
        self.features.swap(i, j)
        self.labels.swap(i, j)
        self.weights.swap(i, j)
      }
    }
  }
}

///|
pub fn DataBatch::slice(self : DataBatch, start : Int, end : Int) -> DataBatch {
  let from = if start < 0 {
    0
  } else if start > self.size() {
    self.size()
  } else {
    start
  }
  let to = if end < from {
    from
  } else if end > self.size() {
    self.size()
  } else {
    end
  }
  let features = Array::makei(to - from, i => {
    copy_vector(self.features[from + i])
  })
  let labels = Array::makei(to - from, i => self.labels[from + i])
  let weights = Array::makei(to - from, i => self.weights[from + i])
  { features, labels, weights }
}

///|
pub struct StreamCounters {
  mut rows : Int
  mut accepted : Int
  mut rejected : Int
  mut positive : Int
  mut negative : Int
}

///|
pub fn StreamCounters::new() -> StreamCounters {
  { rows: 0, accepted: 0, rejected: 0, positive: 0, negative: 0 }
}

///|
pub fn StreamCounters::observe(
  self : StreamCounters,
  accepted : Bool,
  label? : Double = 0.0,
) -> Unit {
  self.rows += 1
  if accepted {
    self.accepted += 1
    if label >= 0.5 {
      self.positive += 1
    } else {
      self.negative += 1
    }
  } else {
    self.rejected += 1
  }
}

///|
pub fn StreamCounters::rows(self : StreamCounters) -> Int {
  self.rows
}

///|
pub fn StreamCounters::accepted(self : StreamCounters) -> Int {
  self.accepted
}

///|
pub fn StreamCounters::rejected(self : StreamCounters) -> Int {
  self.rejected
}

///|
pub fn StreamCounters::acceptance_rate(self : StreamCounters) -> Double {
  if self.rows == 0 {
    0.0
  } else {
    self.accepted.to_double() / self.rows.to_double()
  }
}
