Generic function bodies must satisfy their declared dtype-family bounds: an
unbounded or `Numeric` binder cannot use a float-only operation without a
sufficient `Float` contract. Checked operation-family restrictions now survive
function values and transitive calls. Older checked-context caches and package
shells must be rebuilt.
