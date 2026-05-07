module Std.Tests.Loss.CrossEntropy
import Std.Loss.CrossEntropy (loss)
import Std.Test (assert_true)
def expected_loss_shape_2class(logits: tensor[1, 2, f32], labels: tensor[1, 2, f32]) -> tensor[1, f32] = loss(logits, labels)
def expected_loss_shape_multiclass(logits: tensor[4, 3, f32], labels: tensor[4, 3, f32]) -> tensor[4, f32] = loss(logits, labels)
def test_crossentropy_module_imports() -> unit ! { Test } = { assert_true(true, "Std.Loss.CrossEntropy.loss export resolves") }
def test_crossentropy_typechecks_with_concrete_shapes() -> unit ! { Test } = { assert_true(true, "Std.Loss.CrossEntropy.loss concrete-shape signature type-checks") }
