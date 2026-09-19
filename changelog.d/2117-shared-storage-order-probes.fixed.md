Annotation storage now has test coverage of its shared path. `insert`,
`remove` and `replace` locate a key against a shared borrow and apply that
index after `Arc::make_mut`, which returns a different allocation when the
storage is shared; every existing order test built a fresh `Metadata`, so
that seam never ran. Five probes hold a live co-owner across each write and
assert the deep-copy count, so a probe that stops reaching the seam fails
rather than passing vacuously, and one test pins the allocation figures the
storage comment states. See
[#2117](https://github.com/Chelis-Lang/chelis/issues/2117).
