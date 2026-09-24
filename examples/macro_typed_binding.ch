macro add_one(v) = (fn (record: f32) -> add(record, v))(1.0f32)
def run(record: f32) -> f32 = {
  result: f32 = add_one(record)
  result
}
out = run(10.0f32)
