**Tier B no longer reports cvc5 counterexamples produced by an
out-of-domain `sqrt` value (chelis#1475).** Every exact square-root argument
must first be proved non-negative from independent, sqrt-free total
preconditions. Unsafe or unproved arguments fall through to Tier C instead
of adding a domain assumption, while guarded and algebraically non-negative
arguments retain SMT discharge. Domain authorization and the main query
share the caller's timeout instead of each consuming a full timeout.
