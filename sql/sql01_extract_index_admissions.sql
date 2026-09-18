-- sql/01_extract_index_admissions.sql
-- Step 1: Identify index HF admissions (first in last 12 months)
WITH heart_failure_admissions AS (
  SELECT 
    e.encounter_id,
    e.patient_id,
    e.admission_date,
    e.discharge_date,
    e.primary_diagnosis_code,
    e.discharge_disposition,
    ROW_NUMBER() OVER (
      PARTITION BY e.patient_id 
      ORDER BY e.admission_date DESC
    ) AS admission_rank
  FROM encounters e
  WHERE e.primary_diagnosis_code IN ('I501', 'I502', 'I503', 'I509')  -- HF codes
    AND e.encounter_type = 'Inpatient'
    AND e.discharge_date >= DATE('now', '-12 months')
    AND e.discharge_date < DATE('now')
),
index_admissions AS (
  SELECT 
    ha.*,
    p.gender,
    CAST((JULIANDAY(ha.admission_date) - JULIANDAY(p.date_of_birth)) / 365.25 AS INTEGER) AS age,
    p.race,
    p.insurance_type,
    p.zip_code,
    CASE 
      WHEN p.date_of_death IS NOT NULL AND p.date_of_death <= DATE(ha.discharge_date, '+30 days') 
      THEN 1 
      ELSE 0 
    END AS died_within_30d
  FROM heart_failure_admissions ha
  JOIN patients p ON ha.patient_id = p.patient_id
  WHERE ha.admission_rank = 1  -- Most recent HF admission
),
-- Step 2: Find readmissions within 30 days
readmission_info AS (
  SELECT 
    ia.encounter_id AS index_encounter_id,
    ia.patient_id,
    ia.admission_date AS index_admission,
    ia.discharge_date AS index_discharge,
    MIN(e2.admission_date) AS readmission_date,
    CASE 
      WHEN MIN(e2.admission_date) IS NOT NULL 
      AND JULIANDAY(MIN(e2.admission_date)) - JULIANDAY(ia.discharge_date) <= 30
      THEN 1 
      ELSE 0 
    END AS readmitted_30d,
    JULIANDAY(MIN(e2.admission_date)) - JULIANDAY(ia.discharge_date) AS days_to_readmit
  FROM index_admissions ia
  LEFT JOIN encounters e2 
    ON ia.patient_id = e2.patient_id
    AND e2.admission_date > ia.discharge_date
    AND e2.encounter_type = 'Inpatient'
  GROUP BY ia.encounter_id, ia.patient_id, ia.admission_date, ia.discharge_date
),
-- Step 3: Calculate comorbidities (Elixhauser categories simplified)
comorbidities AS (
  SELECT 
    d.patient_id,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'I10%' THEN 'Hypertension' END) AS has_hypertension,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'E11%' THEN 'Diabetes' END) AS has_diabetes,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'I25%' THEN 'CAD' END) AS has_cad,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'J44%' THEN 'COPD' END) AS has_copd,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'N18%' THEN 'CKD' END) AS has_ckd,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'I48%' THEN 'Afib' END) AS has_afib,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'E66%' THEN 'Obesity' END) AS has_obesity,
    COUNT(DISTINCT CASE WHEN d.diagnosis_code LIKE 'D64%' THEN 'Anemia' END) AS has_anemia,
    COUNT(DISTINCT d.diagnosis_code) AS total_diagnoses
  FROM diagnoses d
  WHERE d.diagnosis_date <= (SELECT discharge_date FROM index_admissions ia WHERE ia.patient_id = d.patient_id LIMIT 1)
  GROUP BY d.patient_id
),
-- Step 4: Get last lab values before discharge
last_labs AS (
  SELECT 
    l.patient_id,
    MAX(CASE WHEN l.lab_test = 'BNP' AND l.lab_date <= ri.index_discharge THEN l.lab_value END) AS last_bnp,
    MAX(CASE WHEN l.lab_test = 'Sodium' AND l.lab_date <= ri.index_discharge THEN l.lab_value END) AS last_sodium,
    MAX(CASE WHEN l.lab_test = 'Creatinine' AND l.lab_date <= ri.index_discharge THEN l.lab_value END) AS last_creatinine,
    MAX(CASE WHEN l.lab_test = 'Hemoglobin' AND l.lab_date <= ri.index_discharge THEN l.lab_value END) AS last_hemoglobin
  FROM labs l
  JOIN readmission_info ri ON l.patient_id = ri.patient_id
  WHERE l.lab_date >= DATE(ri.index_admission, '-7 days')
    AND l.lab_date <= ri.index_discharge
  GROUP BY l.patient_id
),
-- Step 5: Medication adherence metrics
medication_metrics AS (
  SELECT 
    m.patient_id,
    COUNT(DISTINCT CASE WHEN m.medication_class IN ('ACE Inhibitor', 'ARB', 'Beta Blocker') THEN m.medication_name END) AS hf_med_count,
    MAX(m.refill_count) AS max_refill_count,
    COUNT(DISTINCT m.medication_name) AS total_meds
  FROM medications m
  JOIN readmission_info ri ON m.patient_id = ri.patient_id
  WHERE m.prescribe_date >= DATE(ri.index_admission, '-90 days')
    AND m.prescribe_date <= ri.index_discharge
  GROUP BY m.patient_id
),
-- Step 6: Prior utilization
prior_utilization AS (
  SELECT 
    ia.patient_id,
    COUNT(DISTINCT e.encounter_id) AS prior_admissions_6m,
    COUNT(DISTINCT CASE WHEN e.encounter_type = 'ED' THEN e.encounter_id END) AS prior_ed_visits_30d
  FROM index_admissions ia
  LEFT JOIN encounters e 
    ON ia.patient_id = e.patient_id
    AND e.discharge_date >= DATE(ia.admission_date, '-6 months')
    AND e.discharge_date < ia.admission_date
  GROUP BY ia.patient_id
)
-- Final combined dataset
SELECT 
  ri.*,
  ia.gender,
  ia.age,
  ia.race,
  ia.insurance_type,
  ia.zip_code,
  ia.discharge_disposition,
  ia.died_within_30d,
  
  -- Comorbidities
  c.has_hypertension,
  c.has_diabetes,
  c.has_cad,
  c.has_copd,
  c.has_ckd,
  c.has_afib,
  c.has_obesity,
  c.has_anemia,
  c.total_diagnoses,
  
  -- Labs
  ll.last_bnp,
  ll.last_sodium,
  ll.last_creatinine,
  ll.last_hemoglobin,
  
  -- Medications
  mm.hf_med_count,
  mm.max_refill_count,
  mm.total_meds,
  
  -- Prior utilization
  pu.prior_admissions_6m,
  pu.prior_ed_visits_30d,
  
  -- Process measures
  CAST((JULIANDAY(ri.index_discharge) - JULIANDAY(ri.index_admission)) AS INTEGER) AS length_of_stay,
  CASE 
    WHEN CAST(strftime('%w', ri.index_discharge) AS INTEGER) IN (0, 6) 
    THEN 1 ELSE 0 
  END AS weekend_discharge,
  
  -- Social/administrative (simplified)
  CASE 
    WHEN ia.insurance_type IN ('Medicare', 'Medicaid') THEN 'Public'
    ELSE 'Private'
  END AS insurance_category,
  
  CASE 
    WHEN ia.discharge_disposition IN ('Home', 'AMA') THEN 'Home'
    ELSE 'Facility'
  END AS discharge_location
  
FROM readmission_info ri
JOIN index_admissions ia ON ri.patient_id = ia.patient_id
LEFT JOIN comorbidities c ON ri.patient_id = c.patient_id
LEFT JOIN last_labs ll ON ri.patient_id = ll.patient_id
LEFT JOIN medication_metrics mm ON ri.patient_id = mm.patient_id
LEFT JOIN prior_utilization pu ON ri.patient_id = pu.patient_id
WHERE ia.died_within_30d = 0  -- Exclude patients who died (competing risk)
ORDER BY ri.index_discharge DESC;