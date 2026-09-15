Move broad PR package expansion to an exact-head manual dispatch and validate
PR contract acknowledgements without restarting compiler CI. Base retargets
hold a required exact-head/exact-base implementation receipt. Directly changed
all-ignored integration targets can now opt into required complete ignored-suite
execution with exact per-test receipts.
