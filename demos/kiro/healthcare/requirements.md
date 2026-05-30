# Healthcare Drug Dosing System: Requirements

## Drug Dosing Calculation

**WHEN** a drug administration request is received with valid patient weight and dosing rate
**THE SYSTEM SHALL** return a positive dose value

**WHEN** the calculated dose exceeds the maximum safe dose
**THE SYSTEM SHALL** clamp the dose to the maximum safe value

**WHEN** a pediatric patient (age under 12) is dosed
**THE SYSTEM SHALL** apply the age-based scaling factor and return a non-negative dose

**WHEN** glucose readings increase above target
**THE SYSTEM SHALL** produce non-decreasing insulin response

**THE SYSTEM SHALL** keep pacing rate within physiological bounds of 30 to 200 BPM
