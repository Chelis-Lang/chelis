Pull request CI now publishes default-branch-verified candidate receipts that
bind the exact head, synthetic merge, patch identity, and required-check
provenance. These receipts provide trusted input for later evidence reuse;
pull-request code cannot issue or validate its own receipt.
