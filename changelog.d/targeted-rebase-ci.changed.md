Trusted rebases can reuse prior standing-integration and package-expansion
evidence while validating the complete new tree delta. Documentation-only
updates and conflict resolutions take a cheap validation lane; code deltas run
combined-candidate checks and every affected package-targeted test, while
CI-policy, retargeted, unmapped, or uncertain updates fall back to full CI.
