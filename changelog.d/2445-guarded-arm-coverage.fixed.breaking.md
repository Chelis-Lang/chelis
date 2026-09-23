A guarded `match` arm no longer counts toward exhaustiveness, as
spec/04-type-system.md section 2.4 now states: its guard can be `false`, so the
match could run out of arms. A program whose only arm for a constructor is
guarded, such as `| Some(v) if gt(v, 10i64) => 1 | None => 3`, is now rejected
with `NonExhaustiveMatch`; add an unguarded arm for that constructor or a
wildcard. Before, it checked clean and, once guards were evaluated
([#2445](https://github.com/Chelis-Lang/chelis/issues/2445)), would have
failed at run time.
