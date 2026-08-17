///|
/// In-memory feature registry with schema checks, versioning, and bounded history.
pub struct FeatureSchema {
  name : String
  dimension : Int
  minimum : Array[Double]
  maximum : Array[Double]
  required : Bool
}

///|
pub fn FeatureSchema::new(
  name : String,
  dimension : Int,
  minimum? : Array[Double] = [],
  maximum? : Array[Double] = [],
  required? : Bool = true,
) -> FeatureSchema {
  {
    name,
    dimension: if dimension < 0 {
      0
    } else {
      dimension
    },
    minimum: copy_vector(minimum),
    maximum: copy_vector(maximum),
    required,
  }
}

///|
pub fn FeatureSchema::name(self : FeatureSchema) -> String {
  self.name
}

///|
pub fn FeatureSchema::dimension(self : FeatureSchema) -> Int {
  self.dimension
}

///|
pub fn FeatureSchema::required(self : FeatureSchema) -> Bool {
  self.required
}

///|
pub fn FeatureSchema::validate(
  self : FeatureSchema,
  values : Array[Double],
) -> ValidationReport {
  if self.required && values.is_empty() {
    ValidationReport::error("required feature is empty")
  } else if values.length() != self.dimension {
    ValidationReport::error("feature dimension mismatch")
  } else {
    let mut message = ""
    for i in 0..<values.length() {
      if message == "" && (values[i].is_nan() || values[i].is_inf()) {
        message = "feature contains non-finite value"
      }
      if message == "" &&
        i < self.minimum.length() &&
        values[i] < self.minimum[i] {
        message = "feature below schema minimum"
      }
      if message == "" &&
        i < self.maximum.length() &&
        values[i] > self.maximum[i] {
        message = "feature above schema maximum"
      }
    }
    if message == "" {
      ValidationReport::ok()
    } else {
      ValidationReport::error(message)
    }
  }
}

///|
pub struct FeatureRecord {
  key : String
  version : Int
  values : Array[Double]
  timestamp : Int64
  source : String
}

///|
pub fn FeatureRecord::new(
  key : String,
  values : Array[Double],
  timestamp : Int64,
  source? : String = "unknown",
  version? : Int = 1,
) -> FeatureRecord {
  {
    key,
    version: if version < 1 {
      1
    } else {
      version
    },
    values: copy_vector(values),
    timestamp,
    source,
  }
}

///|
pub fn FeatureRecord::key(self : FeatureRecord) -> String {
  self.key
}

///|
pub fn FeatureRecord::version(self : FeatureRecord) -> Int {
  self.version
}

///|
pub fn FeatureRecord::values(self : FeatureRecord) -> Array[Double] {
  copy_vector(self.values)
}

///|
pub fn FeatureRecord::timestamp(self : FeatureRecord) -> Int64 {
  self.timestamp
}

///|
pub fn FeatureRecord::source(self : FeatureRecord) -> String {
  self.source
}

///|
pub fn FeatureRecord::checksum(self : FeatureRecord) -> String {
  snapshot_checksum(self.values.map(value => "\{value}").join(","))
}

///|
pub struct FeatureStore {
  schemas : Map[String, FeatureSchema]
  records : Map[String, Array[FeatureRecord]]
  capacity : Int
  mut writes : Int
  mut reads : Int
  mut misses : Int
  mut rejected : Int
}

///|
pub fn FeatureStore::new(capacity? : Int = 4) -> FeatureStore {
  {
    schemas: {},
    records: {},
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
    writes: 0,
    reads: 0,
    misses: 0,
    rejected: 0,
  }
}

///|
pub fn FeatureStore::register(
  self : FeatureStore,
  schema : FeatureSchema,
) -> Bool {
  if self.schemas.contains(schema.name()) {
    false
  } else {
    self.schemas[schema.name()] = schema
    true
  }
}

///|
pub fn FeatureStore::put(
  self : FeatureStore,
  name : String,
  record : FeatureRecord,
) -> Bool {
  match self.schemas.get(name) {
    None => {
      self.rejected += 1
      false
    }
    Some(schema) => {
      let report = schema.validate(record.values())
      if !report.is_valid() {
        self.rejected += 1
        false
      } else {
        let history = self.records.get(name).unwrap_or([])
        history.push(record)
        if history.length() > self.capacity {
          let _ = history.remove(0)
        }
        self.records[name] = history
        self.writes += 1
        true
      }
    }
  }
}

///|
pub fn FeatureStore::latest(
  self : FeatureStore,
  name : String,
) -> FeatureRecord? {
  self.reads += 1
  match self.records.get(name) {
    None => {
      self.misses += 1
      None
    }
    Some(history) =>
      match history.last() {
        None => {
          self.misses += 1
          None
        }
        Some(record) => Some(record)
      }
  }
}

///|
pub fn FeatureStore::at_version(
  self : FeatureStore,
  name : String,
  version : Int,
) -> FeatureRecord? {
  self.reads += 1
  match self.records.get(name) {
    None => {
      self.misses += 1
      None
    }
    Some(history) => {
      let mut result : FeatureRecord? = None
      for record in history {
        if record.version() == version {
          result = Some(record)
        }
      }
      match result {
        None => self.misses += 1
        Some(_) => ()
      }
      result
    }
  }
}

///|
pub fn FeatureStore::history(
  self : FeatureStore,
  name : String,
) -> Array[FeatureRecord] {
  self.records.get(name).unwrap_or([]).map(record => record)
}

///|
pub fn FeatureStore::schema(
  self : FeatureStore,
  name : String,
) -> FeatureSchema? {
  self.schemas.get(name)
}

///|
pub fn FeatureStore::writes(self : FeatureStore) -> Int {
  self.writes
}

///|
pub fn FeatureStore::reads(self : FeatureStore) -> Int {
  self.reads
}

///|
pub fn FeatureStore::misses(self : FeatureStore) -> Int {
  self.misses
}

///|
pub fn FeatureStore::rejected(self : FeatureStore) -> Int {
  self.rejected
}

///|
pub fn FeatureStore::hit_rate(self : FeatureStore) -> Double {
  if self.reads == 0 {
    1.0
  } else {
    (self.reads - self.misses).to_double() / self.reads.to_double()
  }
}

///|
pub fn FeatureStore::clear(self : FeatureStore) -> Unit {
  self.records.clear()
  self.writes = 0
  self.reads = 0
  self.misses = 0
  self.rejected = 0
}

///|
pub struct FeatureMaterializer {
  schema : FeatureSchema
  defaults : Array[Double]
  mut materialized : Int
  mut fallback : Int
}

///|
pub fn FeatureMaterializer::new(
  schema : FeatureSchema,
  defaults? : Array[Double] = [],
) -> FeatureMaterializer {
  { schema, defaults: copy_vector(defaults), materialized: 0, fallback: 0 }
}

///|
pub fn FeatureMaterializer::materialize(
  self : FeatureMaterializer,
  values : Array[Double],
) -> Array[Double] {
  if self.schema.validate(values).is_valid() {
    self.materialized += 1
    copy_vector(values)
  } else {
    self.fallback += 1
    Array::makei(self.schema.dimension(), i => {
      if i < self.defaults.length() {
        self.defaults[i]
      } else {
        0.0
      }
    })
  }
}

///|
pub fn FeatureMaterializer::materialized(self : FeatureMaterializer) -> Int {
  self.materialized
}

///|
pub fn FeatureMaterializer::fallback(self : FeatureMaterializer) -> Int {
  self.fallback
}

///|
pub fn FeatureMaterializer::reset(self : FeatureMaterializer) -> Unit {
  self.materialized = 0
  self.fallback = 0
}
