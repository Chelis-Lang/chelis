SMT-only property proving now routes nested scalar-gradient properties
independently, including filtered runs, and reports them unsupported at the
intended lowering boundary while retaining fail-closed errors for invalid
sibling declarations. See
[#2264](https://github.com/Chelis-Lang/chelis/issues/2264).
