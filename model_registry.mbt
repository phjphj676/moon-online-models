///|
/// Lightweight in-process model registry for reproducible experiments.
pub struct ModelArtifact {
  name : String
  version : String
  checksum : String
  created_at : String
  mut stage : String
  metrics : Map[String, Double]
} derive(ToJson, FromJson, Debug)

///|
pub fn ModelArtifact::new(
  name : String,
  version : String,
  checksum : String,
  created_at : String,
) -> ModelArtifact {
  { name, version, checksum, created_at, stage: "candidate", metrics: {} }
}

///|
pub fn ModelArtifact::set_stage(self : ModelArtifact, stage : String) -> Unit {
  self.stage = stage
}

///|
pub fn ModelArtifact::add_metric(
  self : ModelArtifact,
  name : String,
  value : Double,
) -> Unit {
  self.metrics[name] = value
}

///|
pub fn ModelArtifact::name(self : ModelArtifact) -> String {
  self.name
}

///|
pub fn ModelArtifact::version(self : ModelArtifact) -> String {
  self.version
}

///|
pub fn ModelArtifact::checksum(self : ModelArtifact) -> String {
  self.checksum
}

///|
pub fn ModelArtifact::stage(self : ModelArtifact) -> String {
  self.stage
}

///|
pub fn ModelArtifact::metric(self : ModelArtifact, name : String) -> Double? {
  self.metrics.get(name)
}

///|
pub fn ModelArtifact::metrics(self : ModelArtifact) -> Map[String, Double] {
  self.metrics.copy()
}

///|
pub struct ModelRegistry {
  artifacts : Map[String, ModelArtifact]
  mut registrations : Int
}

///|
pub fn ModelRegistry::new() -> ModelRegistry {
  { artifacts: {}, registrations: 0 }
}

///|
pub fn ModelRegistry::register(
  self : ModelRegistry,
  artifact : ModelArtifact,
) -> Bool {
  let key = "\{artifact.name()}@\{artifact.version()}"
  if self.artifacts.contains(key) {
    false
  } else {
    self.artifacts[key] = artifact
    self.registrations += 1
    true
  }
}

///|
pub fn ModelRegistry::get(
  self : ModelRegistry,
  name : String,
  version : String,
) -> ModelArtifact? {
  self.artifacts.get("\{name}@\{version}")
}

///|
pub fn ModelRegistry::promote(
  self : ModelRegistry,
  name : String,
  version : String,
  stage : String,
) -> Bool {
  match self.get(name, version) {
    None => false
    Some(artifact) => {
      artifact.set_stage(stage)
      true
    }
  }
}

///|
pub fn ModelRegistry::versions(
  self : ModelRegistry,
  name : String,
) -> Array[String] {
  self.artifacts
  .keys()
  .filter(key => key.has_prefix("\{name}@"))
  .map(key => key.after("@").unwrap_or("").to_owned())
  .to_array()
}

///|
pub fn ModelRegistry::size(self : ModelRegistry) -> Int {
  self.artifacts.length()
}

///|
pub fn ModelRegistry::registrations(self : ModelRegistry) -> Int {
  self.registrations
}

///|
pub fn ModelRegistry::clear(self : ModelRegistry) -> Unit {
  self.artifacts.clear()
  self.registrations = 0
}

///|
pub struct PromotionGate {
  minimum_auc : Double
  maximum_log_loss : Double
  minimum_samples : Int
}

///|
pub fn PromotionGate::new(
  minimum_auc? : Double = 0.5,
  maximum_log_loss? : Double = 1.0,
  minimum_samples? : Int = 100,
) -> PromotionGate {
  {
    minimum_auc,
    maximum_log_loss,
    minimum_samples: if minimum_samples < 0 {
      0
    } else {
      minimum_samples
    },
  }
}

///|
pub fn PromotionGate::accept(
  self : PromotionGate,
  summary : EvaluationSummary,
) -> Bool {
  summary.samples() >= self.minimum_samples.to_double() &&
  summary.auc() >= self.minimum_auc &&
  summary.log_loss() <= self.maximum_log_loss
}
