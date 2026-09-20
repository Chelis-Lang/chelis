`chelis test` no longer re-runs the effect and linearity checkers, re-lowers the
library, and byte-compares two serializations of it inside every test worker.
The parent now hands each worker the payload digest alongside the tempfile, so
the worker can establish that the bytes are the ones the parent wrote and
reconstruct the context directly. The on-disk compiled-context cache, which has
no such channel, keeps the full re-derivation. Part of #2117; chelis#2211.
