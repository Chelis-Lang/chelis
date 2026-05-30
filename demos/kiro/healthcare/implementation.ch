-- c-earchin-surf-emission: v1
-- Domain: Healthcare - drug dosing and medical device safety
-- Compliance: FDA submissions, IEC 62304, pharmaceutical regulators

-- H1-001: Simple dose calculation (weight * rate)
def calculate_dose(weight_kg: f32, dose_mg_per_kg: f32) -> f32 =
  weight_kg * dose_mg_per_kg

@property prop_H1_001 forall(weight_kg: f32, dose_mg_per_kg: f32)
  where weight_kg > 0.0, dose_mg_per_kg > 0.0:
  calculate_dose(weight_kg, dose_mg_per_kg) > 0.0

-- H1-002: Dose ceiling enforcement via min()
def safe_dose(weight_kg: f32, base_dose_mg_per_kg: f32, max_total_mg: f32) -> f32 =
  if (weight_kg * base_dose_mg_per_kg > max_total_mg)
    then max_total_mg
    else weight_kg * base_dose_mg_per_kg

@property prop_H1_002 forall(weight_kg: f32, base_dose_mg_per_kg: f32, max_total_mg: f32)
  where weight_kg > 0.0, base_dose_mg_per_kg > 0.0, max_total_mg > 0.0:
  safe_dose(weight_kg, base_dose_mg_per_kg, max_total_mg) <= max_total_mg

-- H1-003: Age-adjusted dose (piecewise at age 12, non-negative)
def age_adjusted_dose(age_years: f32, adult_dose_mg: f32) -> f32 =
  if (age_years >= 12.0)
    then adult_dose_mg
    else adult_dose_mg * age_years / 12.0

@property prop_H1_003 forall(age_years: f32, adult_dose_mg: f32)
  where age_years >= 0.0, adult_dose_mg > 0.0:
  age_adjusted_dose(age_years, adult_dose_mg) >= 0.0

-- H1-004: Insulin response monotonicity (quantified - Tier B Unknown -> Tier C)
-- Higher glucose should produce non-decreasing insulin dose
def insulin_response(glucose_mg_dl: f32, prev_glucose_mg_dl: f32, target_mg_dl: f32) -> f32 =
  (glucose_mg_dl - target_mg_dl) * 0.1

@property prop_H1_004 forall(glucose_mg_dl: f32, prev_glucose_mg_dl: f32, target_mg_dl: f32)
  where glucose_mg_dl > target_mg_dl, prev_glucose_mg_dl > target_mg_dl, glucose_mg_dl > prev_glucose_mg_dl:
  insulin_response(glucose_mg_dl, prev_glucose_mg_dl, target_mg_dl) >= insulin_response(prev_glucose_mg_dl, prev_glucose_mg_dl, target_mg_dl)

-- H1-005: Pacing rate bounded [30, 200] (clamp - Tier B proves)
def pacing_rate(measured_hr: f32, target_hr: f32) -> f32 =
  if (target_hr < 30.0) then 30.0
    else if (target_hr > 200.0) then 200.0
    else target_hr

@property prop_H1_005 forall(measured_hr: f32, target_hr: f32):
  pacing_rate(measured_hr, target_hr) >= 30.0
