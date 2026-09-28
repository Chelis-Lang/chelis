`chelis eval` and the HIP and Metal targets close four gaps with the C lane.

- `chelis eval` builds, matches, prints and returns a data-type value nested
  past the depth where it used to overflow its stack (a `fold`-built chain of a
  few thousand links). Copying a value, stamping a value that crosses a
  function interface, printing it and converting a root for output now walk it
  from a worklist. Each read of a fold accumulator still copies the whole
  value, so such a fold stays quadratic in its length
  ([#2567](https://github.com/Chelis-Lang/chelis/issues/2567); the copy is
  [#2592](https://github.com/Chelis-Lang/chelis/issues/2592)).
- `chelis eval` accepts a tensor definition whose body reaches `fold` other
  than as its head: in a block tail, through a block binding or through a
  callee. The shared kernel decision now keeps such a definition on the host
  lane before lowering, as the C lane did
  ([#2574](https://github.com/Chelis-Lang/chelis/issues/2574)).
- `--target hip` and `--target metal` keep a definition whose parameter the
  single DAG entry cannot carry (a string, data-type, container, tuple or unit
  parameter) with its authored signature, by taking their host backend as C
  does, instead of emitting a zero-input entry that dropped it
  ([#2575](https://github.com/Chelis-Lang/chelis/issues/2575)).
- The host program emitted as C++ for Metal (`.mm`) and HIP (`.cpp`) declares
  the runtime's emitter-private entry points with C linkage, so a host program
  using `map`, a string concatenation or a consuming container operation links
  against `libchelis_runtime.a`
  ([#2582](https://github.com/Chelis-Lang/chelis/issues/2582)).
