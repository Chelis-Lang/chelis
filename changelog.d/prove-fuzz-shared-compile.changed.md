`chelis prove`'s fuzz tier compiles one program per chunk of samples instead
of one per evaluation. The guard and predicate probes of every sample in a
chunk of attempts, or of a chunk of one shrink step's trials, are declared in
one program that is checked and lowered once, in the Surf, Deep and
assumption-injection lanes alike; chunks grow from one sample to 64. Verdicts,
counterexamples, shrink steps, seeds and error messages are unchanged. See
[#3117](https://github.com/Chelis-Lang/chelis/issues/3117).
