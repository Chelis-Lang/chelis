`chelis prove --tier beacon-only` reaches the Beacon engine again for a property
whose goal pipes into `tensor_to_scalar`, the spelling the formatter and the
`prefer-pipe-operator` lint produce, and for any property whose source defines a
helper function: the extracted scalar graph now keeps a valid declaration table.
`examples/beacon_scalar_range.ch` runs again.

Public-facing text no longer describes the repositories as private or cites
internal planning labels: the install guide, Reef guide, `chelis-std` skill, and
the help for `chelis build`, `eval`, `prove`, and `reef` say what each command
does, and `chelis build` lists Metal. The `chelis` shim no longer suggests
`chelisup install latest` after `+latest`, the `par` rejection and the
redundant-`copy()` lint use plain wording, and the guide's and skill's Surf
examples are canonically formatted, with a test that keeps them so.
