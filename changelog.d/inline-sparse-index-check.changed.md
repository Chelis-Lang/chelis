Compiled `gather` and `scatter` loops compare each selected index with the
axis extent inline. Only an index outside the axis calls the runtime index
check, and the trap and its message do not change. A program that gathers 16.8
million elements retired 0.93G instructions instead of 2.42G. See
[#3735](https://github.com/Chelis-Lang/chelis/issues/3735).
