Large generated scalar functions avoid repeated tree copying during parsing and
desugaring. Block lookahead shares immutable tokens while retaining its own cursor,
and Deep normalization visits children directly instead of cloning their entire
subtrees first. Binding boundaries, metadata and the normalized representation
remain the same.
