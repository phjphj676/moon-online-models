///|
/// Reproducible native benchmark for the dense online-learning path.
fn main {
  let samples = 20000
  let model = @models.AdagradLogisticRegression::new(4, learning_rate=0.15)
  let auc = @models.AucTracker::new()
  for i in 0..<samples {
    let positive = i % 2 == 0
    let features = if positive {
      [1.0, 0.5, 0.0, 0.25]
    } else {
      [-1.0, -0.5, 0.0, -0.25]
    }
    let label = if positive { 1.0 } else { 0.0 }
    model.update(features, label)
    auc.update(model.predict(features), label)
  }
  let weights = model.weights().map(value => "\{value}").join(",")
  println("samples=\{samples}")
  println("steps=\{model.steps()}")
  println("auc=\{auc.auc()}")
  println("weight_l1=\{@models.l1_norm(model.weights())}")
  println("weight_checksum=\{@models.snapshot_checksum(weights)}")
}
