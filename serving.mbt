///|
/// Deterministic model-serving helpers for embedding online models in services.
pub struct PredictionRequest {
  request_id : String
  features : Array[Double]
  timestamp : Int64
  group : String
}

///|
pub fn PredictionRequest::new(
  request_id : String,
  features : Array[Double],
  timestamp : Int64,
  group? : String = "default",
) -> PredictionRequest {
  { request_id, features: copy_vector(features), timestamp, group }
}

///|
pub fn PredictionRequest::request_id(self : PredictionRequest) -> String {
  self.request_id
}

///|
pub fn PredictionRequest::features(self : PredictionRequest) -> Array[Double] {
  copy_vector(self.features)
}

///|
pub fn PredictionRequest::timestamp(self : PredictionRequest) -> Int64 {
  self.timestamp
}

///|
pub fn PredictionRequest::group(self : PredictionRequest) -> String {
  self.group
}

///|
pub struct PredictionResponse {
  request_id : String
  prediction : Double
  probability : Double
  model_version : String
  latency_ms : Double
  accepted : Bool
}

///|
pub fn PredictionResponse::new(
  request_id : String,
  prediction : Double,
  model_version : String,
  latency_ms? : Double = 0.0,
  accepted? : Bool = true,
) -> PredictionResponse {
  {
    request_id,
    prediction,
    probability: clamp_probability(prediction),
    model_version,
    latency_ms: if latency_ms < 0.0 {
      0.0
    } else {
      latency_ms
    },
    accepted,
  }
}

///|
pub fn PredictionResponse::request_id(self : PredictionResponse) -> String {
  self.request_id
}

///|
pub fn PredictionResponse::prediction(self : PredictionResponse) -> Double {
  self.prediction
}

///|
pub fn PredictionResponse::probability(self : PredictionResponse) -> Double {
  self.probability
}

///|
pub fn PredictionResponse::model_version(self : PredictionResponse) -> String {
  self.model_version
}

///|
pub fn PredictionResponse::latency_ms(self : PredictionResponse) -> Double {
  self.latency_ms
}

///|
pub fn PredictionResponse::accepted(self : PredictionResponse) -> Bool {
  self.accepted
}

///|
pub struct LinearEndpoint {
  weights : Array[Double]
  bias : Double
  version : String
  prediction_guard : PredictionGuard
  mut requests : Int
  mut rejected : Int
}

///|
pub fn LinearEndpoint::new(
  weights : Array[Double],
  bias? : Double = 0.0,
  version? : String = "local",
  lower? : Double = 0.0,
  upper? : Double = 1.0,
) -> LinearEndpoint {
  {
    weights: copy_vector(weights),
    bias,
    version,
    prediction_guard: PredictionGuard::new(lower, upper),
    requests: 0,
    rejected: 0,
  }
}

///|
pub fn LinearEndpoint::predict(
  self : LinearEndpoint,
  request : PredictionRequest,
) -> PredictionResponse {
  self.requests += 1
  let features = request.features()
  let dimension = if features.length() < self.weights.length() {
    features.length()
  } else {
    self.weights.length()
  }
  let mut score = self.bias
  for i in 0..<dimension {
    score += features[i] * self.weights[i]
  }
  let prediction = self.prediction_guard.apply(sigmoid(score))
  PredictionResponse::new(request.request_id(), prediction, self.version)
}

///|
pub fn LinearEndpoint::predict_batch(
  self : LinearEndpoint,
  requests : Array[PredictionRequest],
) -> Array[PredictionResponse] {
  requests.map(request => self.predict(request))
}

///|
pub fn LinearEndpoint::version(self : LinearEndpoint) -> String {
  self.version
}

///|
pub fn LinearEndpoint::weights(self : LinearEndpoint) -> Array[Double] {
  copy_vector(self.weights)
}

///|
pub fn LinearEndpoint::requests(self : LinearEndpoint) -> Int {
  self.requests
}

///|
pub fn LinearEndpoint::rejected(self : LinearEndpoint) -> Int {
  self.rejected
}

///|
pub fn LinearEndpoint::reset(self : LinearEndpoint) -> Unit {
  self.requests = 0
  self.rejected = 0
  self.prediction_guard.reset()
}

///|
pub struct RequestBatcher {
  capacity : Int
  requests : Array[PredictionRequest]
  mut flushed : Int
  mut dropped : Int
}

///|
pub fn RequestBatcher::new(capacity : Int) -> RequestBatcher {
  {
    capacity: if capacity < 1 {
      1
    } else {
      capacity
    },
    requests: [],
    flushed: 0,
    dropped: 0,
  }
}

///|
pub fn RequestBatcher::add(
  self : RequestBatcher,
  request : PredictionRequest,
) -> Bool {
  if request.features().is_empty() {
    self.dropped += 1
    false
  } else {
    self.requests.push(request)
    true
  }
}

///|
pub fn RequestBatcher::ready(self : RequestBatcher) -> Bool {
  self.requests.length() >= self.capacity
}

///|
pub fn RequestBatcher::flush(self : RequestBatcher) -> Array[PredictionRequest] {
  let batch = self.requests.map(request => request)
  if !batch.is_empty() {
    self.flushed += 1
  }
  self.requests.clear()
  batch
}

///|
pub fn RequestBatcher::pending(self : RequestBatcher) -> Int {
  self.requests.length()
}

///|
pub fn RequestBatcher::flushed(self : RequestBatcher) -> Int {
  self.flushed
}

///|
pub fn RequestBatcher::dropped(self : RequestBatcher) -> Int {
  self.dropped
}

///|
pub fn RequestBatcher::reset(self : RequestBatcher) -> Unit {
  self.requests.clear()
  self.flushed = 0
  self.dropped = 0
}

///|
pub struct ShadowEvaluator {
  primary_version : String
  shadow_version : String
  mut comparisons : Int
  mut disagreements : Int
  mut absolute_difference : Double
}

///|
pub fn ShadowEvaluator::new(
  primary_version : String,
  shadow_version : String,
) -> ShadowEvaluator {
  {
    primary_version,
    shadow_version,
    comparisons: 0,
    disagreements: 0,
    absolute_difference: 0.0,
  }
}

///|
pub fn ShadowEvaluator::observe(
  self : ShadowEvaluator,
  primary : Double,
  shadow : Double,
  tolerance? : Double = 0.05,
) -> Bool {
  let difference = (primary - shadow).abs()
  self.comparisons += 1
  self.absolute_difference += difference
  let disagreement = difference > tolerance
  if disagreement {
    self.disagreements += 1
  }
  disagreement
}

///|
pub fn ShadowEvaluator::comparisons(self : ShadowEvaluator) -> Int {
  self.comparisons
}

///|
pub fn ShadowEvaluator::disagreements(self : ShadowEvaluator) -> Int {
  self.disagreements
}

///|
pub fn ShadowEvaluator::disagreement_rate(self : ShadowEvaluator) -> Double {
  if self.comparisons == 0 {
    0.0
  } else {
    self.disagreements.to_double() / self.comparisons.to_double()
  }
}

///|
pub fn ShadowEvaluator::mean_absolute_difference(
  self : ShadowEvaluator,
) -> Double {
  if self.comparisons == 0 {
    0.0
  } else {
    self.absolute_difference / self.comparisons.to_double()
  }
}

///|
pub fn ShadowEvaluator::versions(self : ShadowEvaluator) -> (String, String) {
  (self.primary_version, self.shadow_version)
}

///|
pub fn ShadowEvaluator::reset(self : ShadowEvaluator) -> Unit {
  self.comparisons = 0
  self.disagreements = 0
  self.absolute_difference = 0.0
}

///|
pub struct TrafficSplitter {
  buckets : Map[String, Double]
  default_version : String
  mut assigned : Int
}

///|
pub fn TrafficSplitter::new(default_version : String) -> TrafficSplitter {
  { buckets: {}, default_version, assigned: 0 }
}

///|
pub fn TrafficSplitter::set(
  self : TrafficSplitter,
  version : String,
  share : Double,
) -> Unit {
  self.buckets[version] = clamp(share, 0.0, 1.0)
}

///|
pub fn TrafficSplitter::route(self : TrafficSplitter, hash : Int) -> String {
  self.assigned += 1
  let value = (if hash < 0 { -hash } else { hash }) % 10000
  let point = value.to_double() / 10000.0
  let mut cumulative = 0.0
  let mut selected = self.default_version
  for version in self.buckets.keys() {
    cumulative += self.buckets.get(version).unwrap_or(0.0)
    if point < cumulative && selected == self.default_version {
      selected = version
    }
  }
  selected
}

///|
pub fn TrafficSplitter::share(
  self : TrafficSplitter,
  version : String,
) -> Double {
  self.buckets.get(version).unwrap_or(0.0)
}

///|
pub fn TrafficSplitter::assigned(self : TrafficSplitter) -> Int {
  self.assigned
}

///|
pub fn TrafficSplitter::reset(self : TrafficSplitter) -> Unit {
  self.assigned = 0
}

///|
pub struct ServingStats {
  mut requests : Int
  mut successes : Int
  mut failures : Int
  mut total_latency : Double
  mut max_latency : Double
  mut bytes : Int
}

///|
pub fn ServingStats::new() -> ServingStats {
  {
    requests: 0,
    successes: 0,
    failures: 0,
    total_latency: 0.0,
    max_latency: 0.0,
    bytes: 0,
  }
}

///|
pub fn ServingStats::observe(
  self : ServingStats,
  response : PredictionResponse,
  payload_bytes? : Int = 0,
) -> Unit {
  self.requests += 1
  if response.accepted() {
    self.successes += 1
  } else {
    self.failures += 1
  }
  self.total_latency += response.latency_ms()
  if response.latency_ms() > self.max_latency {
    self.max_latency = response.latency_ms()
  }
  self.bytes += if payload_bytes < 0 { 0 } else { payload_bytes }
}

///|
pub fn ServingStats::requests(self : ServingStats) -> Int {
  self.requests
}

///|
pub fn ServingStats::success_rate(self : ServingStats) -> Double {
  if self.requests == 0 {
    1.0
  } else {
    self.successes.to_double() / self.requests.to_double()
  }
}

///|
pub fn ServingStats::mean_latency(self : ServingStats) -> Double {
  if self.requests == 0 {
    0.0
  } else {
    self.total_latency / self.requests.to_double()
  }
}

///|
pub fn ServingStats::max_latency(self : ServingStats) -> Double {
  self.max_latency
}

///|
pub fn ServingStats::bytes(self : ServingStats) -> Int {
  self.bytes
}

///|
pub fn ServingStats::reset(self : ServingStats) -> Unit {
  self.requests = 0
  self.successes = 0
  self.failures = 0
  self.total_latency = 0.0
  self.max_latency = 0.0
  self.bytes = 0
}

///|
pub struct OnlineBatchScorer {
  endpoint : LinearEndpoint
  batcher : RequestBatcher
  stats : ServingStats
}

///|
pub fn OnlineBatchScorer::new(
  weights : Array[Double],
  batch_size? : Int = 32,
  version? : String = "local",
) -> OnlineBatchScorer {
  {
    endpoint: LinearEndpoint::new(weights, version~),
    batcher: RequestBatcher::new(batch_size),
    stats: ServingStats::new(),
  }
}

///|
pub fn OnlineBatchScorer::submit(
  self : OnlineBatchScorer,
  request : PredictionRequest,
) -> Array[PredictionResponse] {
  if !self.batcher.add(request) {
    []
  } else if self.batcher.ready() {
    self.flush()
  } else {
    []
  }
}

///|
pub fn OnlineBatchScorer::flush(
  self : OnlineBatchScorer,
) -> Array[PredictionResponse] {
  let requests = self.batcher.flush()
  let responses = self.endpoint.predict_batch(requests)
  for response in responses {
    self.stats.observe(response)
  }
  responses
}

///|
pub fn OnlineBatchScorer::pending(self : OnlineBatchScorer) -> Int {
  self.batcher.pending()
}

///|
pub fn OnlineBatchScorer::stats(self : OnlineBatchScorer) -> ServingStats {
  self.stats
}

///|
pub fn OnlineBatchScorer::endpoint(self : OnlineBatchScorer) -> LinearEndpoint {
  self.endpoint
}

///|
pub fn OnlineBatchScorer::reset(self : OnlineBatchScorer) -> Unit {
  self.batcher.reset()
  self.endpoint.reset()
  self.stats.reset()
}
