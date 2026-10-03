`chelis test` no longer fails with `could not supervise test suite: Operation not
permitted` when it signals a suite process group whose remaining members are
still exiting, which macOS reports as a permission error. The supervisor retries
that signal for up to two seconds; a denial that lasts longer still fails the
command with the original error.
