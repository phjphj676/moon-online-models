///|
test "dimension-safe vector and matrix boundaries" {
  let empty = DenseVector::new(-3)
  inspect(empty.dimension(), content="0")
  inspect(empty.set(0, 1.0), content="false")
  let left = DenseVector::from_array([1.0, 2.0])
  let right = DenseVector::from_array([3.0])
  inspect(left.dot_checked(right) is None, content="true")
  let identity = Matrix::identity(2)
  inspect(identity.rows(), content="2")
  inspect(identity.cols(), content="2")
  inspect(identity.get(-1, 0) is None, content="true")
  inspect(identity.set(1, 1, 4.0), content="true")
  inspect(identity.is_symmetric(), content="true")
  inspect(
    identity.determinant() > 3.9 && identity.determinant() < 4.1,
    content="true",
  )
}

///|
test "validation and numerical guards" {
  inspect(validate_vector([1.0], 2).is_valid(), content="false")
  inspect(validate_probability(-0.1).is_valid(), content="false")
  inspect(clamp_probability(-100.0) > 0.0, content="true")
  inspect(sigmoid(-100.0) > 0.0, content="true")
  inspect(softplus(100.0) > 99.0, content="true")
  let grad_guard = GradientGuard::new(lower=-1.0, upper=1.0)
  let clipped = grad_guard.apply([-10.0, 0.5, 10.0])
  inspect(
    clipped[0] == -1.0 && clipped[1] == 0.5 && clipped[2] == 1.0,
    content="true",
  )
  inspect(grad_guard.clipped(), content="2")
  let prediction_guard = PredictionGuard::new(0.0, 1.0)
  inspect(prediction_guard.apply(2.0), content="1")
  inspect(prediction_guard.repaired(), content="1")
}

///|
test "losses and weighted accumulators" {
  inspect(loss_kind_catalog().length(), content="7")
  inspect(loss_value(Squared, 3.0, 1.0), content="2")
  inspect(loss_value(Absolute, 3.0, 1.0), content="2")
  inspect(loss_gradient(Huber, 3.0, 1.0, parameter=1.0), content="1")
  let tracker = LossAccumulator::new(LogLoss)
  let _ = tracker.update(0.9, 1.0)
  let _ = tracker.update(0.1, 0.0)
  inspect(tracker.mean() < 0.2, content="true")
  let schedule = LossSchedule::new(1.0, decay=1.0, floor=0.1)
  inspect(schedule.value(), content="1")
  let _ = schedule.advance()
  inspect(schedule.value() > 0.49 && schedule.value() < 0.51, content="true")
}

///|
test "adagrad and softmax learn separable streams" {
  let classifier = AdagradLogisticRegression::new(2, learning_rate=0.2)
  for _ in 0..<80 {
    classifier.update([1.0, 0.0], 1.0)
    classifier.update([0.0, 1.0], 0.0)
  }
  inspect(classifier.steps(), content="160")
  inspect(
    classifier.predict([1.0, 0.0]) > classifier.predict([0.0, 1.0]),
    content="true",
  )
  let softmax = OnlineSoftmaxRegression::new(3, 2, learning_rate=0.2)
  for _ in 0..<30 {
    let _ = softmax.update([1.0, 0.0], 0)
    let _ = softmax.update([0.0, 1.0], 1)
    let _ = softmax.update([-1.0, -1.0], 2)
  }
  inspect(softmax.steps(), content="90")
  inspect(
    softmax.predict_class([1.0, 0.0]).map(value => value == 0).unwrap_or(false),
    content="true",
  )
  inspect(softmax.update([1.0], 99), content="false")
}

///|
test "sparse models and feature hashing boundaries" {
  let sparse = SparseVector::new(4)
  inspect(sparse.dimension(), content="4")
  let invalid_sparse = sparse.set(-1, 2.0)
  inspect(invalid_sparse.nnz(), content="0")
  let positive_sparse = sparse.set(1, 2.0)
  inspect(positive_sparse.get(1), content="2")
  let classifier = SparseAdagradClassifier::new(4, learning_rate=0.2)
  for _ in 0..<40 {
    classifier.update(positive_sparse, 1.0)
    classifier.update(SparseVector::from_dense([0.0, 0.0, 1.0, 0.0]), 0.0)
  }
  inspect(classifier.steps(), content="80")
  inspect(classifier.predict(positive_sparse) > 0.5, content="true")
  let hasher = FeatureHasher::new(8)
  let hashed = hasher.encode(["blue", "green", "blue"])
  inspect(hashed.dimension(), content="8")
  inspect(hashed.nnz() > 0, content="true")
}

///|
test "metrics handle empty and perfect streams" {
  let empty_auc = AucTracker::new()
  inspect(empty_auc.auc(), content="0.5")
  let auc = AucTracker::new()
  auc.update(0.9, 1.0)
  auc.update(0.8, 1.0)
  auc.update(0.2, 0.0)
  auc.update(0.1, 0.0)
  inspect(auc.auc() > 0.99, content="true")
  let confusion = ConfusionMatrix::new()
  confusion.update(0.9, 1.0)
  confusion.update(0.1, 0.0)
  inspect(confusion.accuracy(), content="1")
  inspect(confusion.mcc(), content="1")
  let calibration = CalibrationTracker::new(bin_count=4)
  calibration.update(0.9, 1.0)
  calibration.update(0.1, 0.0)
  inspect(calibration.bins().length(), content="4")
}

///|
test "stream statistics and drift detectors" {
  let moments = RunningMoments::new()
  moments.update(1.0)
  moments.update(3.0)
  inspect(moments.mean(), content="2")
  inspect(moments.variance(), content="2")
  let detector = PageHinkleyDetector::new(threshold=2.0, delta=0.1)
  let _ = detector.update(0.0)
  let _ = detector.update(0.0)
  let _ = detector.update(10.0)
  inspect(detector.count(), content="3")
  let std = Standardizer::new(2)
  let _ = std.update_and_transform([1.0])
  inspect(std.mean()[0] == 1.0 && std.mean()[1] == 0.0, content="true")
  inspect(std.variance().length(), content="2")
}

///|
test "data stream parsing and bounded sampling" {
  let options = CsvOptions::new()
  let parsed = options.parse_line("a,\"b,c\",d")
  inspect(parsed.length(), content="3")
  inspect(parsed[0], content="a")
  inspect(parsed[1], content="b,c")
  inspect(parsed[2], content="d")
  inspect(csv_write_row(["a", "b,c"]), content="a,\"b,c\"")
  let sampler = ReservoirSampler::new(2, seed=42)
  let _ = sampler.observe(1.0)
  let _ = sampler.observe(2.0)
  let _ = sampler.observe(3.0)
  inspect(sampler.sample().length(), content="2")
  let zero = ReservoirSampler::new(0)
  inspect(zero.observe(1.0), content="false")
  inspect(zero.sample().length(), content="0")
}

///|
test "time series ranking and uncertainty" {
  let window = SequenceWindow::new(3)
  window.push(1.0)
  window.push(2.0)
  window.push(3.0)
  window.push(4.0)
  inspect(window.size(), content="3")
  inspect(window.mean(), content="3")
  let ar = OnlineAR::new(2, learning_rate=0.1)
  let _ = ar.update(1.0)
  let _ = ar.update(2.0)
  let _ = ar.update(3.0)
  inspect(ar.observations(), content="3")
  let interval = ConformalInterval::new(capacity=2, confidence=0.9)
  interval.observe(1.0, 2.0)
  interval.observe(1.0, 1.5)
  inspect(interval.radius() >= 0.5, content="true")
  let ranking = RankingMetrics::new(max_k=3)
  ranking.observe([0.0, 2.0, 1.0])
  inspect(ranking.mrr() > 0.0, content="true")
}

///|
test "feature store rejects invalid dimensions" {
  let schema = FeatureSchema::new("profile", 2, minimum=[0.0, 0.0], maximum=[
    1.0, 1.0,
  ])
  let store = FeatureStore::new(capacity=2)
  inspect(store.register(schema), content="true")
  inspect(
    store.put("profile", FeatureRecord::new("u1", [0.2], 1)),
    content="false",
  )
  inspect(
    store.put("profile", FeatureRecord::new("u1", [0.2, 0.8], 2)),
    content="true",
  )
  inspect(
    store.latest("profile").map(record => record.key()).unwrap_or("") == "u1",
    content="true",
  )
  inspect(store.at_version("profile", 99) is None, content="true")
  inspect(store.rejected(), content="1")
}

///|
test "monitoring, fairness, serving and lifecycle" {
  let monitor = ModelMonitor::new("ctr")
  monitor.add_rule(AlertRule::greater("high-loss", "loss", 0.5))
  let alerts = monitor.observe(MetricSnapshot::new("loss", 0.8, 10), 10)
  inspect(alerts.length(), content="1")
  let fairness = FairnessMonitor::new()
  fairness.observe("a", 0.9, 1.0)
  fairness.observe("b", 0.1, 1.0)
  inspect(fairness.max_recall_gap() > 0.9, content="true")
  let endpoint = LinearEndpoint::new([2.0], version="v1")
  let response = endpoint.predict(PredictionRequest::new("r1", [1.0], 10))
  inspect(response.model_version(), content="v1")
  inspect(response.probability() > 0.5, content="true")
  let deployment = Deployment::new("ctr", "v1", 0)
  inspect(deployment.transition(Staging, 1), content="true")
  inspect(deployment.transition(Active, 2), content="false")
  inspect(deployment.transition(Canary, 2), content="true")
  inspect(deployment.set_traffic(0.2), content="true")
}

///|
test "audit trail, privacy and cost sensitivity" {
  let trail = AuditTrail::new(capacity=2)
  let record = AuditRecord::new("e1", "tester", Train, "m", "v1", 1)
  inspect(trail.append(record), content="true")
  inspect(trail.append(record), content="false")
  inspect(trail.duplicates(), content="1")
  let budget = PrivacyBudget::new(1.0)
  inspect(budget.consume(0.6), content="true")
  inspect(budget.consume(0.6), content="false")
  inspect(budget.rejected(), content="1")
  let costs = ClassCost::new(false_negative=4.0, false_positive=1.0)
  inspect(costs.balanced_threshold() < 0.5, content="true")
  let evaluator = CostSensitiveEvaluator::new(costs~)
  let _ = evaluator.observe(0.9, 1.0, 0.5)
  inspect(evaluator.count(), content="1")
}
