module Std.Tests.Nn.Conv
import Std.Nn.Conv (conv1d, conv2d_small)
import Std.Test (assert_true)
def expected_conv1d_shape(x: tensor[1, 4, 1, 16, f32], k: tensor[8, 4, 1, 3, f32]) -> tensor[1, 8, 1, 14, f32] = conv1d(x, k)
def expected_conv2d_small_shape(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv2d_small(x, k)
def test_conv_module_imports() -> unit ! { Test } = { assert_true(true, "Std.Nn.Conv exports resolve") }
def test_conv_typechecks_with_concrete_shapes() -> unit ! { Test } = { assert_true(true, "Std.Nn.Conv concrete-shape signatures type-check") }
