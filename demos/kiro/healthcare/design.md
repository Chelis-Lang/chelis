# Healthcare Drug Dosing System: Design

## Core Functions

```chelis
def calculate_dose(weight_kg: f32, dose_mg_per_kg: f32) -> f32
def safe_dose(weight_kg: f32, base_dose: f32, max_total: f32) -> f32
def age_adjusted_dose(age_years: f32, adult_dose: f32) -> f32
def insulin_response(glucose: f32, prev_glucose: f32, target: f32) -> f32
def pacing_rate(measured_hr: f32, target_hr: f32) -> f32
```

## Architecture

The dosing system validates inputs, computes doses using weight-based
calculations, enforces safety ceilings, and applies age-based adjustments
for pediatric patients.
