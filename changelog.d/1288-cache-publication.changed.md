Restrict shared durable cache publication to its two concrete library and
standard-library context payloads. Their payload formats advance to library
version 12 and standard-library version 16 under versioned cache keys; older
entries miss before payload decoding. Compiled contexts use their separate
version-18 envelope for both disk caches and worker handoffs.
