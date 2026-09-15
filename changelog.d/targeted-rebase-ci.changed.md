Trusted rebases can reuse prior standing-integration and package-expansion
evidence while validating the complete new tree delta. Documentation-only
patches with documentation-only updates take a cheap validation lane;
code-bearing candidates run current Rust units and every affected
package-targeted test, while CI-policy, retargeted, unmapped, or uncertain
updates fall back to full CI before expensive fan-out.
