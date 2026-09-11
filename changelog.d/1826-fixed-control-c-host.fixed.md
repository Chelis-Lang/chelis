Fixed-control `dropout` now reaches ordinary C source, API, and CLI builds
without losing its source execution plan (part of
[#1764](https://github.com/Chelis-Lang/chelis/issues/1764) and
[#1192](https://github.com/Chelis-Lang/chelis/issues/1192)). An opaque host carrier
binds concrete generic-helper graphs to their private forward/replay metadata
through ownership verification. Direct entries keep the public four-argument
ABI; host helpers use the existing invocation-local RNG frame. Runtime controls
and GPU targets remain typed rejections.
