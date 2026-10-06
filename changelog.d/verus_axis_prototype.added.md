Chelis now uses a shared checked axis plan in type inference, IR lowering,
validation, and permutation differentiation. An experimental Verus runner
proves its four executable axis functions, and a separate Vermilion runner
kernel-checks all four against the same contracts in Lean.
Opt-in Kani harnesses and a resumable mutation runner compare those contracts
with compiler integration tests, requiring independent violation witnesses
before proof failures receive detection credit.
