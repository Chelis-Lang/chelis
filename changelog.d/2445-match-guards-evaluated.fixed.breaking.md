A `match` arm guard (`| pattern if condition => body`) is now evaluated in
every execution lane, as [04-PAT-2] states: after the arm's pattern matches,
the guard runs with the pattern's bindings, and a `false` guard passes control
to the next arm. Before, `chelis eval` selected a guarded arm whenever its
pattern matched, and compiled C dropped guarded and variable arms, let the last
wildcard or `Some` arm win, or selected the first constructor arm whose pattern
matched. Programs with guards can return different values. A guard whose
evaluation traps now makes the `match` trap in both lanes. See
[#2445](https://github.com/Chelis-Lang/chelis/issues/2445).
