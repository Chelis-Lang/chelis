Generic function bodies must satisfy their declared dtype-family bounds: an
unbounded or `Numeric` binder cannot use a float-only operation without a
sufficient `Float` contract. Omitting a signature does not publish an inferred
generic admission contract; local inference holes must bind within their
enclosing declaration or be justified by its declared bounds. Checked
operation-family restrictions now survive function values and transitive
calls. The specialized `reduce_window_*` shape path consumes those same
contracts: `reduce_window_mean` requires `Float`, while window sum, max and min
require `Numeric`. Older checked-context caches and package shells must be
rebuilt.
