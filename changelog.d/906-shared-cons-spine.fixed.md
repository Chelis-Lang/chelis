Compiler readers of canonical `Cons`/`Nil` lists now use one iterative
traversal across evaluation, type and shape checks, resugaring, and lowering.
Malformed list tails remain rejected, and the shared reader avoids native
stack growth proportional to a flat list's length. See
[#906](https://github.com/Chelis-Lang/chelis/issues/906).
