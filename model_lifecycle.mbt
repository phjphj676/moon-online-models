///|
/// Explicit deployment state machine used by registry and serving layers.
pub(all) enum DeploymentState {
  Draft
  Staging
  Canary
  Active
  Paused
  RolledBack
  Retired
} derive(Debug, Eq)

///|
pub fn deployment_state_catalog() -> Array[DeploymentState] {
  [Draft, Staging, Canary, Active, Paused, RolledBack, Retired]
}

///|
pub struct Deployment {
  model : String
  version : String
  mut state : DeploymentState
  created_at : Int64
  mut updated_at : Int64
  mut traffic : Double
}

///|
pub fn Deployment::new(
  model : String,
  version : String,
  timestamp : Int64,
) -> Deployment {
  {
    model,
    version,
    state: Draft,
    created_at: timestamp,
    updated_at: timestamp,
    traffic: 0.0,
  }
}

///|
pub fn Deployment::model(self : Deployment) -> String {
  self.model
}

///|
pub fn Deployment::version(self : Deployment) -> String {
  self.version
}

///|
pub fn Deployment::state(self : Deployment) -> DeploymentState {
  self.state
}

///|
pub fn Deployment::traffic(self : Deployment) -> Double {
  self.traffic
}

///|
pub fn Deployment::transition(
  self : Deployment,
  next : DeploymentState,
  timestamp : Int64,
) -> Bool {
  let allowed = match (self.state, next) {
    (Draft, Staging) => true
    (Staging, Canary) => true
    (Canary, Active) => true
    (Canary, Paused) => true
    (Active, Paused) => true
    (Active, RolledBack) => true
    (Paused, Active) => true
    (Paused, RolledBack) => true
    (RolledBack, Draft) => true
    (Active, Retired) => true
    (Paused, Retired) => true
    _ => false
  }
  if allowed {
    self.state = next
    self.updated_at = timestamp
    if next != Canary {
      self.traffic = if next == Active { 1.0 } else { 0.0 }
    }
    true
  } else {
    false
  }
}

///|
pub fn Deployment::set_traffic(self : Deployment, share : Double) -> Bool {
  if self.state == Canary || self.state == Active {
    self.traffic = clamp(share, 0.0, 1.0)
    true
  } else {
    false
  }
}

///|
pub struct RollbackPolicy {
  minimum_accuracy : Double
  maximum_loss : Double
  maximum_error_rate : Double
}

///|
pub fn RollbackPolicy::new(
  minimum_accuracy? : Double = 0.5,
  maximum_loss? : Double = 1.0,
  maximum_error_rate? : Double = 0.5,
) -> RollbackPolicy {
  {
    minimum_accuracy: clamp(minimum_accuracy, 0.0, 1.0),
    maximum_loss: if maximum_loss < 0.0 {
      0.0
    } else {
      maximum_loss
    },
    maximum_error_rate: clamp(maximum_error_rate, 0.0, 1.0),
  }
}

///|
pub fn RollbackPolicy::should_rollback(
  self : RollbackPolicy,
  accuracy : Double,
  loss : Double,
  error_rate : Double,
) -> Bool {
  accuracy < self.minimum_accuracy ||
  loss > self.maximum_loss ||
  error_rate > self.maximum_error_rate
}

///|
pub struct DeploymentManager {
  deployments : Map[String, Deployment]
  policy : RollbackPolicy
  mut transitions : Int
  mut rollbacks : Int
}

///|
pub fn DeploymentManager::new(
  policy? : RollbackPolicy = RollbackPolicy::new(),
) -> DeploymentManager {
  { deployments: {}, policy, transitions: 0, rollbacks: 0 }
}

///|
pub fn DeploymentManager::create(
  self : DeploymentManager,
  deployment : Deployment,
) -> Bool {
  let key = "\{deployment.model()}@\{deployment.version()}"
  if self.deployments.contains(key) {
    false
  } else {
    self.deployments[key] = deployment
    true
  }
}

///|
pub fn DeploymentManager::transition(
  self : DeploymentManager,
  model : String,
  version : String,
  next : DeploymentState,
  timestamp : Int64,
) -> Bool {
  let key = "\{model}@\{version}"
  match self.deployments.get(key) {
    None => false
    Some(deployment) => {
      let changed = deployment.transition(next, timestamp)
      if changed {
        self.transitions += 1
        if next == RolledBack {
          self.rollbacks += 1
        }
      }
      changed
    }
  }
}

///|
pub fn DeploymentManager::get(
  self : DeploymentManager,
  model : String,
  version : String,
) -> Deployment? {
  self.deployments.get("\{model}@\{version}")
}

///|
pub fn DeploymentManager::healthy(
  self : DeploymentManager,
  accuracy : Double,
  loss : Double,
  error_rate : Double,
) -> Bool {
  !self.policy.should_rollback(accuracy, loss, error_rate)
}

///|
pub fn DeploymentManager::transitions(self : DeploymentManager) -> Int {
  self.transitions
}

///|
pub fn DeploymentManager::rollbacks(self : DeploymentManager) -> Int {
  self.rollbacks
}

///|
pub fn DeploymentManager::clear(self : DeploymentManager) -> Unit {
  self.deployments.clear()
  self.transitions = 0
  self.rollbacks = 0
}

///|
pub struct CanaryExperiment {
  name : String
  baseline : String
  candidate : String
  target_samples : Int
  mut baseline_metric : Double
  mut candidate_metric : Double
  mut samples : Int
}

///|
pub fn CanaryExperiment::new(
  name : String,
  baseline : String,
  candidate : String,
  target_samples? : Int = 100,
) -> CanaryExperiment {
  {
    name,
    baseline,
    candidate,
    target_samples: if target_samples < 1 {
      1
    } else {
      target_samples
    },
    baseline_metric: 0.0,
    candidate_metric: 0.0,
    samples: 0,
  }
}

///|
pub fn CanaryExperiment::observe(
  self : CanaryExperiment,
  baseline : Double,
  candidate : Double,
) -> Unit {
  self.baseline_metric += baseline
  self.candidate_metric += candidate
  self.samples += 1
}

///|
pub fn CanaryExperiment::complete(self : CanaryExperiment) -> Bool {
  self.samples >= self.target_samples
}

///|
pub fn CanaryExperiment::baseline(self : CanaryExperiment) -> Double {
  if self.samples == 0 {
    0.0
  } else {
    self.baseline_metric / self.samples.to_double()
  }
}

///|
pub fn CanaryExperiment::candidate(self : CanaryExperiment) -> Double {
  if self.samples == 0 {
    0.0
  } else {
    self.candidate_metric / self.samples.to_double()
  }
}

///|
pub fn CanaryExperiment::improvement(self : CanaryExperiment) -> Double {
  self.candidate() - self.baseline()
}

///|
pub fn CanaryExperiment::samples(self : CanaryExperiment) -> Int {
  self.samples
}

///|
pub fn CanaryExperiment::winner(self : CanaryExperiment) -> String {
  if self.candidate() >= self.baseline() {
    self.candidate
  } else {
    self.baseline
  }
}
