///|
/// A small, versioned envelope for application-owned model snapshots.
pub struct SnapshotEnvelope {
  schema : String
  model : String
  version : String
  payload : String
  checksum : String
} derive(ToJson, FromJson, Debug)

///|
pub fn SnapshotEnvelope::new(
  schema : String,
  model : String,
  version : String,
  payload : String,
  checksum : String,
) -> SnapshotEnvelope {
  { schema, model, version, payload, checksum }
}

///|
pub fn SnapshotEnvelope::schema(self : SnapshotEnvelope) -> String {
  self.schema
}

///|
pub fn SnapshotEnvelope::model(self : SnapshotEnvelope) -> String {
  self.model
}

///|
pub fn SnapshotEnvelope::version(self : SnapshotEnvelope) -> String {
  self.version
}

///|
pub fn SnapshotEnvelope::payload(self : SnapshotEnvelope) -> String {
  self.payload
}

///|
pub fn SnapshotEnvelope::checksum(self : SnapshotEnvelope) -> String {
  self.checksum
}

///|
pub fn SnapshotEnvelope::is_compatible(
  self : SnapshotEnvelope,
  schema : String,
  model : String,
) -> Bool {
  self.schema == schema && self.model == model
}

///|
pub fn SnapshotEnvelope::to_json_string(self : SnapshotEnvelope) -> String {
  @json.to_json(self).stringify(indent=2)
}

///|
pub fn snapshot_checksum(payload : String) -> String {
  let hash = payload.hash()
  if hash < 0 {
    "-\{-hash}"
  } else {
    "\{hash}"
  }
}

///|
pub fn make_snapshot(
  model : String,
  version : String,
  payload : String,
) -> SnapshotEnvelope {
  SnapshotEnvelope::new(
    "moon-online-models/v1",
    model,
    version,
    payload,
    snapshot_checksum(payload),
  )
}

///|
pub fn verify_snapshot(envelope : SnapshotEnvelope) -> Bool {
  snapshot_checksum(envelope.payload()) == envelope.checksum()
}

///|
pub struct SnapshotCatalog {
  snapshots : Map[String, SnapshotEnvelope]
  mut writes : Int
}

///|
pub fn SnapshotCatalog::new() -> SnapshotCatalog {
  { snapshots: {}, writes: 0 }
}

///|
pub fn SnapshotCatalog::put(
  self : SnapshotCatalog,
  key : String,
  snapshot : SnapshotEnvelope,
) -> Bool {
  if !verify_snapshot(snapshot) {
    false
  } else {
    self.snapshots[key] = snapshot
    self.writes += 1
    true
  }
}

///|
pub fn SnapshotCatalog::get(
  self : SnapshotCatalog,
  key : String,
) -> SnapshotEnvelope? {
  self.snapshots.get(key)
}

///|
pub fn SnapshotCatalog::keys(self : SnapshotCatalog) -> Array[String] {
  self.snapshots.keys().to_array()
}

///|
pub fn SnapshotCatalog::size(self : SnapshotCatalog) -> Int {
  self.snapshots.length()
}

///|
pub fn SnapshotCatalog::writes(self : SnapshotCatalog) -> Int {
  self.writes
}

///|
pub fn SnapshotCatalog::clear(self : SnapshotCatalog) -> Unit {
  self.snapshots.clear()
  self.writes = 0
}
