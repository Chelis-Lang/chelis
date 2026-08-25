# Reef editor schemas

These JSON Schema files describe the syntax and table shape of each Reef document version.

They do not verify paths, package identity, compiler compatibility, network sources, hashes, or archive bytes.

Reef parsing remains the validation authority. Editor validation is advisory.

## Taplo example

Add the applicable directive as the first comment in an editor-only document or template:

```toml
#:schema ./docs/schemas/reef/manifest-v2.schema.json
```

Use `manifest-v1.schema.json` only for a schema-1 manifest.

Use this directive for a lock document:

```toml
#:schema ./docs/schemas/reef/lock-v1.schema.json
```

If the package is outside this repository, use a path that resolves from the document location.
