Local tensor ascriptions now retain their runtime extent obligations through
evaluation, C lowering, transforms, and cached library composition. Helper
result labels also remain distinct from unrelated authored binders after
rank-polymorphic inlining.
