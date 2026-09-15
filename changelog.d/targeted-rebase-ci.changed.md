Trusted rebases can reuse prior standing-integration and package-expansion
evidence while validating the complete new tree delta. Documentation-only
patches with documentation-only updates take a cheap validation lane;
code-bearing candidates run current Rust units and every affected
package-targeted test. Exact Cargo target ownership is checked before fan-out;
ambiguous targets, CI-policy changes, retargets, unmapped paths, or uncertain
updates fall back to full CI even when the PR patch is documentation-only.
