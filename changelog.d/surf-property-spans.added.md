`chelis prove` accepts verified requirement spans for Surf properties, including
Reef-linked properties, and reports original requirement locations in text and
JSON. Versioned manifests verify source and generated-artifact hashes, byte
ranges, and property mappings before property results are emitted. Legacy Deep
source manifests also verify their source hashes and derive display text from
the original bytes; stale or malformed manifests are input errors.
