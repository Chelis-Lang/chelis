module Std.Tests.Loss.Metrics
import Std.Loss.Metrics (accuracy, perplexity)
import Std.Test (assert_close)
def test_accuracy_perfect_is_one() -> unit ! { Test } = {
  logits = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32), cast(0.0, f32)]], cast(3, int64), cast(0.0, f32))
  labels = to_tensor([cast(0, int64), cast(1, int64)])
  out = accuracy(logits, labels)
  assert_close(out, cast(1.0, f32), cast(0.0, f32), "accuracy with argmax==label every row is 1.0 (2/2)")
}
def test_accuracy_zero_when_no_match() -> unit ! { Test } = {
  logits = pad_sequences_to([[cast(0.0, f32), cast(1.0, f32), cast(0.0, f32)], [cast(1.0, f32), cast(0.0, f32), cast(0.0, f32)]], cast(3, int64), cast(0.0, f32))
  labels = to_tensor([cast(0, int64), cast(1, int64)])
  out = accuracy(logits, labels)
  assert_close(out, cast(0.0, f32), cast(0.0, f32), "accuracy with no argmax matching label is 0.0 (0/2)")
}
def test_accuracy_half_when_one_of_two_match() -> unit ! { Test } = {
  logits = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32), cast(0.0, f32)]], cast(3, int64), cast(0.0, f32))
  labels = to_tensor([cast(0, int64), cast(0, int64)])
  out = accuracy(logits, labels)
  assert_close(out, cast(0.5, f32), cast(0.0, f32), "accuracy with 1-of-2 argmax matching label is 0.5 (1/2)")
}
def test_perplexity_zero_loss_is_one() -> unit ! { Test } = assert_close(perplexity(cast(0.0, f32)), cast(1.0, f32), cast(0.000001, f32), "perplexity(0) == exp(0) == 1.0")
def test_perplexity_log_three_is_three() -> unit ! { Test } = assert_close(perplexity(cast(1.0986123, f32)), cast(3.0, f32), cast(0.0001, f32), "perplexity(ln 3) == exp(ln 3) == 3.0")
