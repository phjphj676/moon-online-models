///|
test "RLS model learning" {
  let rls = RLS::new(2)
  rls.update([1.0, 0.0], 2.0)
  rls.update([0.0, 1.0], 3.0)
  rls.update([1.0, 1.0], 5.0)
  
  let pred = rls.predict([1.0, 0.0])
  let is_close = (pred - 2.0) < 0.5 && (pred - 2.0) > -0.5
  inspect(is_close, content="true")
}

///|
test "FTRL model learning" {
  let ftrl = FTRL::new(2, alpha=1.0, l1=0.0, l2=0.0)
  for _ in 0..<100 {
    ftrl.update([0.0, 0.0], 0.0)
    ftrl.update([1.0, 0.0], 0.0)
    ftrl.update([0.0, 1.0], 0.0)
    ftrl.update([1.0, 1.0], 1.0)
  }
  let p = ftrl.predict([1.0, 1.0])
  inspect(p > 0.5, content="true")
}

///|
test "Standardizer" {
  let std = Standardizer::new(1)
  let _ = std.update_and_transform([10.0])
  let _ = std.update_and_transform([20.0])
  let out = std.update_and_transform([30.0])
  inspect(out[0] > 0.0, content="true")
}

///|
test "MetricsTracker" {
  let metrics = MetricsTracker::new()
  metrics.update(0.9, 1.0)
  metrics.update(0.1, 0.0)
  let logloss = metrics.log_loss()
  let mse = metrics.mse()
  inspect(mse < 0.1, content="true")
  inspect(logloss < 0.2, content="true")
}
