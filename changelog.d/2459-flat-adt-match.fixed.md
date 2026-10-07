The C backend lowers wide, guard-free constructor matches as one flat tag
dispatch, avoiding stack exhaustion and quadratic generated C size. See
[#2459](https://github.com/Chelis-Lang/chelis/issues/2459).
