module Std.Datetime.Clock
export (MonotonicInstant, clock_now, monotonic_now, monotonic_until)
import Std.Datetime (Instant, Duration, instant_from_unix, duration)
-- Std.Datetime.Clock: the host clocks, governed by [05-OP-73]. It is the only
-- module of the Std.Datetime family that reads the host. Any code that reads a
-- clock, here or through the builtins directly, carries `IO`. Each read is one [05-OP-75]
-- builtin and carries `IO`. A failed read fails under the builtin's name,
-- `clock_wall_read: io: <detail>` or `clock_monotonic_read: io: <detail>`,
-- because a library definition cannot relabel a builtin's failure.
--
-- A monotonic reading is opaque: its origin is unspecified, so the only
-- datetime function over one is the exact elapsed time between two
-- readings, and nothing converts one to an Instant.
@opaque
type MonotonicInstant =
  | MonotonicInstant { second: i64, nanosecond: i64 }
-- [05-OP-75] admits exactly the instant range's unix seconds, so
-- `instant_from_unix` accepts every reading.
def clock_now() -> Instant ! { IO } = {
  (second, nanosecond) = clock_wall_read()
  instant_from_unix(second, nanosecond)
}
def monotonic_now() -> MonotonicInstant ! { IO } = {
  (second, nanosecond) = clock_monotonic_read()
  MonotonicInstant { second, nanosecond }
}
-- Both readings' seconds lie in the [05-OP-75] range, so their difference is
-- under 2^40 and `duration` normalizes it without failing.
def monotonic_until(a: MonotonicInstant, b: MonotonicInstant) -> Duration = duration(sub(b.second, a.second), sub(b.nanosecond, a.nanosecond))
