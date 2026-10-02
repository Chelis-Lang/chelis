module Std.Tests.Datetime.Clock
import Std.Datetime (Instant, Duration, instant_to_string, parse_instant, duration, duration_negate, duration_gte, duration_lte)
import Std.Datetime.Clock (MonotonicInstant, clock_now, monotonic_now, monotonic_until)
import Std.Test (assert_true)
-- These read the real host clocks, so each asserts only what holds for every
-- reading; exact readings from an injected clock are pinned in the
-- evaluator's tests.
def test_clock_now_round_trips_through_text() -> unit ! { Test, IO } = {
  now = clock_now()
  assert_true(eq(parse_instant(instant_to_string(now)), now), "a wall reading's canonical text parses back to it")
}
def test_monotonic_until_a_reading_and_itself_is_zero() -> unit ! { Test, IO } = {
  mark = monotonic_now()
  assert_true(eq(monotonic_until(mark, mark), duration(0i64, 0i64)), "no time elapses from a reading to itself")
}
def test_monotonic_until_successive_readings_is_not_negative() -> unit ! { Test, IO } = {
  first = monotonic_now()
  second = monotonic_now()
  _ = assert_true(duration_gte(monotonic_until(first, second), duration(0i64, 0i64)), "the monotonic clock never runs backwards")
  assert_true(duration_lte(monotonic_until(second, first), duration(0i64, 0i64)), "the reversed pair is never positive")
}
def test_monotonic_until_is_antisymmetric() -> unit ! { Test, IO } = {
  first = monotonic_now()
  second = monotonic_now()
  assert_true(eq(monotonic_until(second, first), duration_negate(monotonic_until(first, second))), "reversing the readings negates the duration")
}
