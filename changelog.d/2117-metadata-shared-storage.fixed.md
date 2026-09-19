Deep AST annotation storage is now shared behind a reference count, so cloning
a Deep tree copies one pointer per node instead of deep-copying every node's
annotations. A write still copies through `Arc::make_mut`, so the sharing is
invisible. See [#2117](https://github.com/Chelis-Lang/chelis/issues/2117).
