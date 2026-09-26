Downstream shells may add an optional Nix verification job that rebuilds the
shell with atoll's `chelis2nix` and requires the bytes of the shell's chelisup
build. `spec/design/shell_repo_contract.md` §2 states its conditions: the
chelisup lane stays the gate, the job runs only on pushes to `main`, and it
takes the compiler only by substitution and never builds it.
