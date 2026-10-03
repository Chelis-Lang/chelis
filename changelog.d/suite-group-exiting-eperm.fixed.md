On macOS, `chelis test` no longer fails with `could not supervise test suite:
Operation not permitted` when it signals a suite process group whose leader has
exited but is not yet reaped, or whose other members are still exiting. macOS
refuses that signal with a permission error. The supervisor now reaps an exited
leader at once and retries the signal for up to two seconds. A refusal that
lasts longer still fails the command with the original error.
