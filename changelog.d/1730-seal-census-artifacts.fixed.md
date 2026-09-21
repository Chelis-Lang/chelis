Seal the cache-publication census's exact compiled artifacts into its receipt
directory, so later legitimate Cargo builds cannot invalidate a completed
binding witness by replacing files in the shared mutable target cache.
