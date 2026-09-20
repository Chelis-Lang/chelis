The ownership verifier's `inconsistent live owners` error now names the owners
the two paths disagree about, with the source binding names lowering kept for
them, instead of printing two owner-id sets and nothing else:

```text
block b1 in `roots` is reached with inconsistent live owners: live only on this
path: %2[carry]; live only on the path already verified: none (3 live here, 2
earlier)
```

The previous form, `has inconsistent live owners: {0, 1, 5} vs {0, 1}`, left no
way to map an owner id back to authored source. An owner with no recorded names
still prints as `%2`, and the unit and block identifiers are unchanged.

This is the most frequent ownership-verifier message in downstream field data.
Whether the programs that trigger it should be rejected at all is separate and
remains open.
