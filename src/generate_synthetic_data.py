# src/generate_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3
import csv

def generate_synthetic_patients(n_patients=10000, seed=42):
    """Generate synthetic patient data"""
    np.random.seed(seed)
    random.seed(seed)
    
    # Demographics
    genders = ['M', 'F']
    races = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
    insurance_types = ['Medicare', 'Medicaid', 'Private', 'Self-pay']
    
    patients = []
    for i in range(n_patients):
        patient_id = f"P{i:06d}"
        gender = random.choice(genders)
        age = np.random.normal(72, 12)
        age = max(40, min(age, 95))
        
        patients.append({
            'patient_id': patient_id,
            'gender': gender,
            'date_of_birth': datetime.now() - timedelta(days=int(age*365.25)),
            'race': random.choice(races),
            'insurance_type': random.choice(insurance_types),
            'zip_code': f"{random.randint(10000, 99999)}"
        })
    
    return pd.DataFrame(patients)

def generate_encounters(patients, n_encounters_per_patient=3):
    """Generate synthetic encounter data"""
    encounters = []
    hf_codes = ['I501', 'I502', 'I503', 'I509']  # Heart failure ICD-10 codes
    
    for _, patient in patients.iterrows():
        patient_id = patient['patient_id']
        age = (datetime.now() - patient['date_of_birth']).days / 365.25
        
        # Generate 1-6 encounters per patient
        n_encounters = random.randint(1, n_encounters_per_patient)
        
        for i in range(n_encounters):
            encounter_id = f"E{len(encounters):08d}"
            
            # Admission date (within last 3 years)
            days_ago = random.randint(0, 3*365)
            admission_date = datetime.now() - timedelta(days=days_ago)
            
            # Length of stay (skewed distribution)
            los = np.random.exponential(5)
            los = max(1, min(los, 30))
            discharge_date = admission_date + timedelta(days=int(los))
            
            # 30% chance this is a heart failure admission
            is_hf = random.random() < 0.3
            primary_dx = random.choice(hf_codes) if is_hf else random.choice(['I10', 'E11', 'I25', 'J44'])
            
            encounters.append({
                'encounter_id': encounter_id,
                'patient_id': patient_id,
                'admission_date': admission_date,
                'discharge_date': discharge_date,
                'encounter_type': 'Inpatient',
                'primary_diagnosis_code': primary_dx,
                'discharge_disposition': random.choice(['Home', 'SNF', 'Hospice', 'AMA']),
                'is_heart_failure': is_hf
            })
    
    return pd.DataFrame(encounters)

def generate_diagnoses(encounters, n_diagnoses_per_encounter=5):
    """Generate secondary diagnoses for encounters"""
    diagnoses = []
    
    # Common comorbidities for HF patients
    comorbidity_codes = {
        'Hypertension': ['I10'],
        'Diabetes': ['E11'],
        'CAD': ['I25'],
        'COPD': ['J44'],
        'CKD': ['N18'],
        'Atrial Fibrillation': ['I48'],
        'Obesity': ['E66'],
        'Anemia': ['D64']
    }
    
    diagnosis_id = 0
    for _, encounter in encounters.iterrows():
        patient_id = encounter['patient_id']
        encounter_id = encounter['encounter_id']
        admission_date = encounter['admission_date']
        
        # Add primary diagnosis
        diagnoses.append({
            'diagnosis_id': diagnosis_id,
            'patient_id': patient_id,
            'encounter_id': encounter_id,
            'diagnosis_code': encounter['primary_diagnosis_code'],
            'diagnosis_date': admission_date,
            'diagnosis_type': 'Primary'
        })
        diagnosis_id += 1
        
        # Add 0-8 secondary diagnoses
        n_secondary = random.randint(0, 8)
        for _ in range(n_secondary):
            comorbidity = random.choice(list(comorbidity_codes.keys()))
            dx_code = random.choice(comorbidity_codes[comorbidity])
            
            diagnoses.append({
                'diagnosis_id': diagnosis_id,
                'patient_id': patient_id,
                'encounter_id': encounter_id,
                'diagnosis_code': dx_code,
                'diagnosis_date': admission_date + timedelta(days=random.randint(0, 3)),
                'diagnosis_type': 'Secondary'
            })
            diagnosis_id += 1
    
    return pd.DataFrame(diagnoses)

def generate_labs(encounters):
    """Generate lab results"""
    labs = []
    lab_id = 0
    
    lab_tests = {
        'BNP': (300, 2000, 100),  # mean, std, min
        'Sodium': (138, 4, 125),
        'Creatinine': (1.2, 0.8, 0.3),
        'Potassium': (4.2, 0.5, 3.0),
        'Hemoglobin': (12.5, 2.0, 7.0)
    }
    
    for _, encounter in encounters.iterrows():
        patient_id = encounter['patient_id']
        encounter_id = encounter['encounter_id']
        admission_date = encounter['admission_date']
        discharge_date = encounter['discharge_date']
        
        # Generate 1-3 lab draws per encounter
        n_draws = random.randint(1, 3)
        for draw in range(n_draws):
            draw_date = admission_date + timedelta(days=draw)
            
            for test_name, (mean, std, min_val) in lab_tests.items():
                # Simulate lab value with some trend
                base_value = np.random.normal(mean, std)
                base_value = max(min_val, base_value)
                
                # Heart failure patients tend to have higher BNP
                if test_name == 'BNP' and encounter['is_heart_failure']:
                    base_value *= np.random.uniform(1.5, 3.0)
                
                labs.append({
                    'lab_id': lab_id,
                    'patient_id': patient_id,
                    'encounter_id': encounter_id,
                    'lab_test': test_name,
                    'lab_value': round(base_value, 2),
                    'lab_date': draw_date
                })
                lab_id += 1
    
    return pd.DataFrame(labs)

def generate_medications(encounters):
    """Generate medication data"""
    medications = []
    med_id = 0
    
    hf_meds = [
        ('Lisinopril', 'ACE Inhibitor'),
        ('Losartan', 'ARB'),
        ('Metoprolol', 'Beta Blocker'),
        ('Furosemide', 'Diuretic'),
        ('Spironolactone', 'Aldosterone Antagonist')
    ]
    
    other_meds = [
        ('Insulin', 'Diabetes'),
        ('Metformin', 'Diabetes'),
        ('Atorvastatin', 'Statin'),
        ('Warfarin', 'Anticoagulant'),
        ('Aspirin', 'Anti-platelet')
    ]
    
    for _, encounter in encounters.iterrows():
        patient_id = encounter['patient_id']
        encounter_id = encounter['encounter_id']
        admission_date = encounter['admission_date']
        
        # HF patients get HF meds
        med_list = hf_meds if encounter['is_heart_failure'] else other_meds
        
        # 1-4 medications per encounter
        n_meds = random.randint(1, 4)
        selected_meds = random.sample(med_list, min(n_meds, len(med_list)))
        
        for med_name, med_class in selected_meds:
            prescribe_date = admission_date + timedelta(days=random.randint(0, 2))
            
            medications.append({
                'medication_id': med_id,
                'patient_id': patient_id,
                'encounter_id': encounter_id,
                'medication_name': med_name,
                'medication_class': med_class,
                'prescribe_date': prescribe_date,
                'refill_count': random.randint(0, 3)
            })
            med_id += 1
    
    return pd.DataFrame(medications)

def save_to_sqlite(df_dict, db_path='data/raw/ehr_synthetic.db'):
    """Save all dataframes to SQLite database"""
    conn = sqlite3.connect(db_path)
    
    for table_name, df in df_dict.items():
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Saved {len(df)} rows to {table_name}")
    
    conn.close()

def main():
    print("Generating synthetic EHR data...")
    
    # Generate data
    patients_df = generate_synthetic_patients(n_patients=5000)
    encounters_df = generate_encounters(patients_df, n_encounters_per_patient=3)
    diagnoses_df = generate_diagnoses(encounters_df)
    labs_df = generate_labs(encounters_df)
    medications_df = generate_medications(encounters_df)
    
    # Add death dates for ~20% of patients
    patients_df['date_of_death'] = None
    deceased_patients = patients_df.sample(frac=0.2)
    for idx in deceased_patients.index:
        death_date = datetime.now() - timedelta(days=random.randint(30, 365*2))
        patients_df.at[idx, 'date_of_death'] = death_date
    
    # Save to files
    data_dict = {
        'patients': patients_df,
        'encounters': encounters_df,
        'diagnoses': diagnoses_df,
        'labs': labs_df,
        'medications': medications_df
    }
    
    save_to_sqlite(data_dict)
    
    # Also save as CSV for easy viewing
    for name, df in data_dict.items():
        df.to_csv(f'data/raw/{name}.csv', index=False)
    
    print("\nData generation complete!")
    print(f"Patients: {len(patients_df)}")
    print(f"Encounters: {len(encounters_df)} (HF: {encounters_df['is_heart_failure'].sum()})")
    print(f"Diagnoses: {len(diagnoses_df)}")
    print(f"Labs: {len(labs_df)}")
    print(f"Medications: {len(medications_df)}")
    
    return data_dict

if __name__ == "__main__":
    main()