Check fixed JSON-number adapters at each source, report, count, and extent field
that owns their numeric domain. Reusing one in an unrelated field or generic
wrapper fails discovery, even when the underlying numeric declaration is already
present in the census. Transparent type aliases preserve the field check.
