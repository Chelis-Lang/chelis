module Std.Tests.Init.Xavier
import Std.Init.Xavier (sample)
import Std.Test (assert_true)
def expected_sample_shape(template: tensor[32, 128, f32], gain: f32) -> tensor[32, 128, f32] ! { Random } = sample(template, gain)
def expected_sample_seeded(template: tensor[32, 128, f32], gain: f32) -> tensor[32, 128, f32] = with seed(42i64) { sample(template, gain) }
def expected_sample_seeded_distinct(template: tensor[32, 128, f32], gain: f32) -> tensor[32, 128, f32] = with seed(7i64) { sample(template, gain) }
def test_xavier_module_imports() -> unit ! { Test } = { assert_true(true, "Std.Init.Xavier exports resolve") }
def test_xavier_typechecks_with_concrete_shape() -> unit ! { Test } = { assert_true(true, "Std.Init.Xavier.sample concrete-shape signature type-checks") }
def test_xavier_with_seed_call_site_compiles() -> unit ! { Test } = { assert_true(true, "Std.Init.Xavier.sample with seed(...) call site compiles at rank-2 shape") }
