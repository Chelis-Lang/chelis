The Nix flake checks `lockParity`, `crate2nixGraph`, `compilerBehavior`, and
`lane-check` now read source views with only the files that they use. Before,
they read the full flake source, 74 MB with the documentation. So a
documentation edit no longer changes a check derivation. See [#3909](https://github.com/Chelis-Lang/chelis/issues/3909).
