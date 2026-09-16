Targeted rebase CI can now read the trusted prior-candidate receipts it needs
to select an incremental validation lane. Previously the verifier lacked a
GitHub token and failed closed to full CI for every eligible rebase.
