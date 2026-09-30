A core-fragment eval/C parity receipt runs a pinned corpus of programs through
both `chelis eval` and the compiled C lane and compares their complete
observations byte for byte, and their traps by exit status and complete
diagnostic. The corpus is a versioned manifest that records every case's
expected outcome and every excluded file's reason, so a run cannot pass
vacuously. It is a manual gate, documented in
[`docs/manual_gates.md`](../docs/manual_gates.md); its contract is
[`spec/design/core_fragment_parity_corpus.md`](../spec/design/core_fragment_parity_corpus.md).
See [#2102](https://github.com/Chelis-Lang/chelis/issues/2102).
