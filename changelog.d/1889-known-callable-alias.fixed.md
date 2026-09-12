Native C lowering preserves statically known local aliases of named tensor
helpers, including alias chains and lexical shadowing. Named-axis result labels
and authored runtime extent guards follow the captured definition and signature.
Call arguments retain caller scope and eager execution even when unused.
This bounded #1889 repair does not add dynamic runtime callables or a new
dimension/provenance representation.
