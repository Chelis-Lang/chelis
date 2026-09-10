Fixed-control source evaluation now applies dropout at every active float dtype,
validates its same-dtype rate before drawing, and preserves accepted empty and
dead-value draws. First-order input gradients replay the actual forward mask
without advancing the ambient stream; nested seed handlers preserve parent
state on errors. Ordinary, prepared, and contextual evaluator paths share this
execution plan. Legacy Dag-only APIs and wire/cache formats are unchanged;
runtime-rate AD, random vmap, higher-order AD, general UniformLike numerics, and
compiled dropout remain outside this bounded repair of
[#1295](https://github.com/Chelis-Lang/chelis/issues/1295).
