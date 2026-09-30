`chelis build` now compiles a program that reads a loop's result again inside a
later loop. A list built by `map`, `filter`, `scan`, `fold` or `flat_map` and
then passed by value to a named definition from inside another loop's callback
was rejected by ownership verification as reaching a block with inconsistent
live owners, even though `chelis check` accepted it and `chelis eval` ran it
correctly. The loop result is now copied for that call, as [04-LIN-5] requires
of any use that preserves another live use. A program that already built still
builds and still produces the same result. See
[#2781](https://github.com/Chelis-Lang/chelis/issues/2781).
