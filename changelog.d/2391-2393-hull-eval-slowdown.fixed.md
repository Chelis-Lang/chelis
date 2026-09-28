`chelis check`, `chelis test` and the evaluator no longer slow down by up to two
orders of magnitude on large recursive programs. Three costs introduced in
0.18.7 are removed without changing any classification, kernel choice or name
resolution: the static-controls classifier re-walked every shared callee along
every call path and was re-run for each definition at every call site
([#2391](https://github.com/Chelis-Lang/chelis/issues/2391)); an application
under an execution exclusion re-planned its callee's kernel every time
([#2392](https://github.com/Chelis-Lang/chelis/issues/2392)); and a short or
import-qualified definition name was resolved by scanning the whole linked
program on each use ([#2393](https://github.com/Chelis-Lang/chelis/issues/2393)).
On the Hull test suite, files that took 85 to over 899 seconds on 0.18.11 now
finish in under 20 seconds.
