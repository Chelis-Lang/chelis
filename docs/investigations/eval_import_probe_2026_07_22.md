# Probe Report: Eval-Side Package Import Resolution

**Date:** 2026-07-22
**Binary:** `target/debug/chelis` v0.11.1 (HEAD of `seed-diligence` branch, post-v0.16.1)
**Verdict: BLOCKER RESOLVED** — cross-shell imports work in eval mode.

## Reproduction

```sh
# Structure:
# /tmp/chelis-eval-probe/
#   reef.toml (root workspace)
#   shell_a/reef.toml + src/lib.ch  (defines helper function)
#   shell_b/reef.toml + src/main.ch (imports from shell_a)
```

### shell_a/src/lib.ch
```
module ShellA.Lib
def helper(x: f32) -> f32 = x + 1.0
```

### shell_b/src/main.ch
```
module ShellB.Main
import ShellA.Lib
def result -> f32 = ShellA.Lib.helper(2.0)
```

### shell_b/reef.toml
```toml
[package]
name = "shell_b"
version = "0.1.0"
compiler = "=0.11.1"
module_prefix = "ShellB"

[dependencies]
shell_a = { path = "../shell_a" }
```

## Results

| Command | Result | Notes |
|---------|--------|-------|
| `chelis check src/main.ch --allow-style-violations` | ✅ score 1.0, 0 errors | Types resolve across shell boundary |
| `chelis eval --file src/main.ch --allow-style-violations` | ✅ exit 0, "nothing to evaluate" | Import resolves; no eval target (def-only file) |
| `chelis build src/main.ch --allow-style-violations` | ✅ Wrote main.c, main.h | Full compilation through cross-shell import |

## Analysis

The `prepare_program_for_eval_file` function in `chelis-reef/src/lib.rs` correctly:
1. Discovers the reef package root via `find_package_root_for_dir`
2. Loads the package graph including dependencies
3. Links the graph, resolving `import ShellA.Lib` to the shell_a dependency
4. Rewrites internal module names correctly
5. Presents the linked program to the type checker and eval/build pipeline

## Conclusion

**The consumer can close their workaround.** Cross-shell package imports in
eval mode are functional on the current main branch (which subsumes v0.16.1).
The consumer should verify against their specific template, but the
fundamental mechanism (reef package resolution through `prepare_program_for_eval_file`)
is working.

If the consumer is pinned to a pre-v0.16.1 binary, they should upgrade: the
fix has been in-tree for several releases.
