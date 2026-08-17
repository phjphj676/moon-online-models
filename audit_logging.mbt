///|
/// Auditable metadata for model updates, data lineage, and privacy budgets.
pub(all) enum AuditAction {
  Train
  Predict
  Promote
  Rollback
  Reject
  Snapshot
} derive(Debug, Eq)

///|
pub fn audit_action_catalog() -> Array[AuditAction] {
  [Train, Predict, Promote, Rollback, Reject, Snapshot]
}

///|
pub struct AuditRecord {
  event_id : String
  actor : String
  action : AuditAction
  model : String
  version : String
  timestamp : Int64
  detail : String
  success : Bool
}

///|
pub fn AuditRecord::new(
  event_id : String,
  actor : String,
  action : AuditAction,
  model : String,
  version : String,
  timestamp : Int64,
  detail? : String = "",
  success? : Bool = true,
) -> AuditRecord {
  { event_id, actor, action, model, version, timestamp, detail, success }
}

///|
pub fn AuditRecord::event_id(self : AuditRecord) -> String {
  self.event_id
}

///|
pub fn AuditRecord::actor(self : AuditRecord) -> String {
  self.actor
}

///|
pub fn AuditRecord::action(self : AuditRecord) -> AuditAction {
  self.action
}

///|
pub fn AuditRecord::model(self : AuditRecord) -> String {
  self.model
}

///|
pub fn AuditRecord::version(self : AuditRecord) -> String {
  self.version
}

///|
pub fn AuditRecord::timestamp(self : AuditRecord) -> Int64 {
  self.timestamp
}

///|
pub fn AuditRecord::detail(self : AuditRecord) -> String {
  self.detail
}

///|
pub fn AuditRecord::success(self : AuditRecord) -> Bool {
  self.success
}

///|
pub struct AuditTrail {
  capacity : Int
  records : Array[AuditRecord]
  ids : Map[String, Bool]
  mut accepted : Int
  mut duplicates : Int
  mut evicted : Int
}

///|
pub fn AuditTrail::new(capacity? : Int = 2048) -> AuditTrail {
  {
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
    records: [],
    ids: {},
    accepted: 0,
    duplicates: 0,
    evicted: 0,
  }
}

///|
pub fn AuditTrail::append(self : AuditTrail, record : AuditRecord) -> Bool {
  if self.ids.contains(record.event_id()) {
    self.duplicates += 1
    false
  } else {
    self.records.push(record)
    self.ids[record.event_id()] = true
    self.accepted += 1
    if self.records.length() > self.capacity {
      let removed = self.records.remove(0)
      self.ids.remove(removed.event_id())
      self.evicted += 1
    }
    true
  }
}

///|
pub fn AuditTrail::size(self : AuditTrail) -> Int {
  self.records.length()
}

///|
pub fn AuditTrail::records(self : AuditTrail) -> Array[AuditRecord] {
  self.records.map(record => record)
}

///|
pub fn AuditTrail::find(self : AuditTrail, event_id : String) -> AuditRecord? {
  for record in self.records {
    if record.event_id() == event_id {
      return Some(record)
    }
  }
  None
}

///|
pub fn AuditTrail::count_action(self : AuditTrail, action : AuditAction) -> Int {
  self.records.count_if(record => record.action() == action)
}

///|
pub fn AuditTrail::failed(self : AuditTrail) -> Int {
  self.records.count_if(record => !record.success())
}

///|
pub fn AuditTrail::accepted(self : AuditTrail) -> Int {
  self.accepted
}

///|
pub fn AuditTrail::duplicates(self : AuditTrail) -> Int {
  self.duplicates
}

///|
pub fn AuditTrail::evicted(self : AuditTrail) -> Int {
  self.evicted
}

///|
pub fn AuditTrail::clear(self : AuditTrail) -> Unit {
  self.records.clear()
  self.ids.clear()
  self.accepted = 0
  self.duplicates = 0
  self.evicted = 0
}

///|
pub struct LineageNode {
  dataset : String
  source : String
  schema : String
  row_count : Int
  checksum : String
  timestamp : Int64
}

///|
pub fn LineageNode::new(
  dataset : String,
  source : String,
  schema : String,
  row_count : Int,
  checksum : String,
  timestamp : Int64,
) -> LineageNode {
  {
    dataset,
    source,
    schema,
    row_count: if row_count < 0 {
      0
    } else {
      row_count
    },
    checksum,
    timestamp,
  }
}

///|
pub fn LineageNode::dataset(self : LineageNode) -> String {
  self.dataset
}

///|
pub fn LineageNode::source(self : LineageNode) -> String {
  self.source
}

///|
pub fn LineageNode::schema(self : LineageNode) -> String {
  self.schema
}

///|
pub fn LineageNode::row_count(self : LineageNode) -> Int {
  self.row_count
}

///|
pub fn LineageNode::checksum(self : LineageNode) -> String {
  self.checksum
}

///|
pub fn LineageNode::timestamp(self : LineageNode) -> Int64 {
  self.timestamp
}

///|
pub struct DataLineage {
  nodes : Map[String, LineageNode]
  edges : Map[String, Array[String]]
  mut registrations : Int
}

///|
pub fn DataLineage::new() -> DataLineage {
  { nodes: {}, edges: {}, registrations: 0 }
}

///|
pub fn DataLineage::register(self : DataLineage, node : LineageNode) -> Bool {
  if self.nodes.contains(node.dataset()) {
    false
  } else {
    self.nodes[node.dataset()] = node
    self.registrations += 1
    true
  }
}

///|
pub fn DataLineage::connect(
  self : DataLineage,
  input : String,
  output : String,
) -> Bool {
  if !self.nodes.contains(input) || !self.nodes.contains(output) {
    false
  } else {
    let dependencies = self.edges.get(output).unwrap_or([])
    if dependencies.contains(input) {
      false
    } else {
      dependencies.push(input)
      self.edges[output] = dependencies
      true
    }
  }
}

///|
pub fn DataLineage::node(self : DataLineage, dataset : String) -> LineageNode? {
  self.nodes.get(dataset)
}

///|
pub fn DataLineage::inputs(
  self : DataLineage,
  dataset : String,
) -> Array[String] {
  self.edges.get(dataset).unwrap_or([]).map(value => value)
}

///|
pub fn DataLineage::upstream(
  self : DataLineage,
  dataset : String,
) -> Array[String] {
  let visited : Map[String, Bool] = Map([])
  let output : Array[String] = []
  fn walk(
    lineage : DataLineage,
    current : String,
    visited : Map[String, Bool],
    output : Array[String],
  ) -> Unit {
    for input in lineage.inputs(current) {
      if !visited.contains(input) {
        visited[input] = true
        output.push(input)
        walk(lineage, input, visited, output)
      }
    }
  }
  walk(self, dataset, visited, output)
  output
}

///|
pub fn DataLineage::registrations(self : DataLineage) -> Int {
  self.registrations
}

///|
pub fn DataLineage::clear(self : DataLineage) -> Unit {
  self.nodes.clear()
  self.edges.clear()
  self.registrations = 0
}

///|
pub struct PrivacyBudget {
  total : Double
  mut remaining_budget : Double
  mut queries : Int
  mut rejected : Int
}

///|
pub fn PrivacyBudget::new(epsilon : Double) -> PrivacyBudget {
  let safe = if epsilon < 0.0 { 0.0 } else { epsilon }
  { total: safe, remaining_budget: safe, queries: 0, rejected: 0 }
}

///|
pub fn PrivacyBudget::consume(self : PrivacyBudget, cost : Double) -> Bool {
  if cost < 0.0 || cost > self.remaining_budget {
    self.rejected += 1
    false
  } else {
    self.remaining_budget -= cost
    self.queries += 1
    true
  }
}

///|
pub fn PrivacyBudget::total(self : PrivacyBudget) -> Double {
  self.total
}

///|
pub fn PrivacyBudget::remaining(self : PrivacyBudget) -> Double {
  self.remaining_budget
}

///|
pub fn PrivacyBudget::spent(self : PrivacyBudget) -> Double {
  self.total - self.remaining_budget
}

///|
pub fn PrivacyBudget::queries(self : PrivacyBudget) -> Int {
  self.queries
}

///|
pub fn PrivacyBudget::rejected(self : PrivacyBudget) -> Int {
  self.rejected
}

///|
pub fn PrivacyBudget::exhausted(self : PrivacyBudget) -> Bool {
  self.remaining_budget <= 0.0
}

///|
pub fn PrivacyBudget::reset(self : PrivacyBudget) -> Unit {
  self.remaining_budget = self.total
  self.queries = 0
  self.rejected = 0
}

///|
pub struct ReproducibilityManifest {
  model : String
  version : String
  source_checksum : String
  data_checksum : String
  seed : Int
  parameters : Map[String, String]
}

///|
pub fn ReproducibilityManifest::new(
  model : String,
  version : String,
  source_checksum : String,
  data_checksum : String,
  seed : Int,
) -> ReproducibilityManifest {
  { model, version, source_checksum, data_checksum, seed, parameters: {} }
}

///|
pub fn ReproducibilityManifest::set(
  self : ReproducibilityManifest,
  key : String,
  value : String,
) -> Unit {
  self.parameters[key] = value
}

///|
pub fn ReproducibilityManifest::model(self : ReproducibilityManifest) -> String {
  self.model
}

///|
pub fn ReproducibilityManifest::version(
  self : ReproducibilityManifest,
) -> String {
  self.version
}

///|
pub fn ReproducibilityManifest::source_checksum(
  self : ReproducibilityManifest,
) -> String {
  self.source_checksum
}

///|
pub fn ReproducibilityManifest::data_checksum(
  self : ReproducibilityManifest,
) -> String {
  self.data_checksum
}

///|
pub fn ReproducibilityManifest::seed(self : ReproducibilityManifest) -> Int {
  self.seed
}

///|
pub fn ReproducibilityManifest::parameter(
  self : ReproducibilityManifest,
  key : String,
) -> String {
  self.parameters.get(key).unwrap_or("")
}

///|
pub fn ReproducibilityManifest::parameter_count(
  self : ReproducibilityManifest,
) -> Int {
  self.parameters.length()
}

///|
pub fn ReproducibilityManifest::fingerprint(
  self : ReproducibilityManifest,
) -> String {
  let text = "\{self.model}:\{self.version}:\{self.source_checksum}:\{self.data_checksum}:\{self.seed}:\{self.parameters.length()}"
  snapshot_checksum(text)
}
