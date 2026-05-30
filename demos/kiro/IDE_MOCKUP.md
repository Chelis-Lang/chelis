# Kiro IDE Integration Mockup

## Requirements View (requirements.md)

In the Kiro IDE, requirements.md displays with verification status:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ requirements.md                                    [5/5 ✓ verified]  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  # Healthcare Drug Dosing System: Requirements                               │
│                                                                              │
│  ## Drug Dosing Calculation                                                  │
│                                                                              │
│  ✓ **WHEN** a drug administration request is received...                     │
│    **THE SYSTEM SHALL** return a positive dose value                         │
│    └─ SMT proved (cvc5, QF_NRA, 47ms)                                       │
│                                                                              │
│  ✓ **WHEN** the calculated dose exceeds the maximum...                       │
│    **THE SYSTEM SHALL** clamp the dose to the maximum safe value             │
│    └─ SMT proved (cvc5, QF_NRA, 31ms)                                       │
│                                                                              │
│  ✓ **WHEN** a pediatric patient (age under 12) is dosed                      │
│    **THE SYSTEM SHALL** apply the age-based scaling factor...                │
│    └─ SMT proved (cvc5, QF_NRA, 28ms)                                       │
│                                                                              │
│  ✓ **WHEN** glucose readings increase above target                           │
│    **THE SYSTEM SHALL** produce non-decreasing insulin response              │
│    └─ SMT proved (cvc5, QF_NRA, 19ms)                                       │
│                                                                              │
│  ✓ **THE SYSTEM SHALL** keep pacing rate within bounds                       │
│    └─ SMT proved (cvc5, QF_NRA, 22ms)                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Hover Tooltip

Hovering over a verified requirement shows:

```
┌─────────────────────────────────────────┐
│ ✓ PROVED by SMT                            │
├─────────────────────────────────────────┤
│ Solver: cvc5 1.3.1                         │
│ Logic: QF_NRA                              │
│ Duration: 47ms                             │
│ Function: calculate_dose                   │
│ Postcondition: weight * rate > 0           │
│ Preconditions: weight > 0, rate > 0        │
└─────────────────────────────────────────┘
```

## Failure View

For a failing property:

```
┌─────────────────────────────────────────┐
│ ✗ DISPROVED by SMT                          │
├─────────────────────────────────────────┤
│ Counterexample:                             │
│   weight = 0.0001                           │
│   rate = 0.0001                             │
│   Result: 0.00000001 (not > threshold)      │
│                                             │
│ Function: buggy_dose                        │
│ Expected: result > 0.001                    │
└─────────────────────────────────────────┘
```

## MCP Integration Flow

```
Kiro IDE                          chelis-tide MCP Server
   │                                      │
   │── tools/call: chelis_verify_spec ───▶│
   │   {spec_directory: "./"}              │
   │                                      │── read .ch files
   │                                      │── verify_source()
   │                                      │── cvc5 proves each property
   │◀── VerificationResult ─────────────│
   │   {properties: [...], summary: ...}   │
   │                                      │
   │── Render gutter icons ────────────▶│
   │   ✓ green = proved                    │
   │   ○ yellow = fuzz validated            │
   │   ✗ red = disproved                   │
```

## Status Icons

| Icon | Meaning | Tier |
|------|---------|------|
| ✓ (green) | Formally proved | SMT (cvc5) |
| ○ (yellow) | Statistically validated | Fuzz (100 samples) |
| ✗ (red) | Disproved with counterexample | SMT or Fuzz |
| - (gray) | Inconclusive / not yet verified | - |
