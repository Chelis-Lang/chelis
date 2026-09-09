Discover compiler wire types through all public serialization exports, including
imported definitions, aliases, private fields, and concrete generic instances.
Retain explicit ownership of binary-cache candidates and bind the defining crate
artifacts to actual codec execution. Missing imports, unresolved generic exports,
and unsupported serializer shapes fail the discovery check.
The verifier also checks the compiled Python crate for binding-local serialized
type definitions, so a private manifest copy requires shared-schema ownership.
