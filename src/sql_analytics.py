"""
Advanced SQL Analytics Engine
Supports CTEs, Window Functions, Analytical Queries
"""

import pandas as pd
import sqlite3
from typing import List, Dict, Optional
import json
from datetime import datetime


class SQLAnalytics:
    """
    Advanced SQL query engine with analytical functions
    """
    
    def __init__(self, db_path: str = 'data/raw/ehr_synthetic.db'):
        self.db_path = db_path
        self.conn = None
        self.query_history = []
        
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        print(f"✅ Connected to {self.db_path}")
        
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute SQL query and return results"""
        if self.conn is None:
            self.connect()
        
        try:
            if params:
                result = pd.read_sql_query(query, self.conn, params=params)
            else:
                result = pd.read_sql_query(query, self.conn)
        except sqlite3.ProgrammingError as exc:
            if "same thread" not in str(exc):
                raise
            self.connect()
            if params:
                result = pd.read_sql_query(query, self.conn, params=params)
            else:
                result = pd.read_sql_query(query, self.conn)
            
        try:
            # Log query
            self.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'rows_returned': len(result)
            })
            
            return result
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str = None) -> pd.DataFrame:
        """Get information about tables in database"""
        if table_name:
            query = f"PRAGMA table_info({table_name})"
            return self.execute_query(query)
        else:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            return self.execute_query(query)
    
    # ==================== ANALYTICAL QUERIES ====================
    
    def patient_cohort_analysis(self) -> pd.DataFrame:
        """
        Cohort analysis with CTEs and window functions
        """
        query = """
        WITH patient_encounters AS (
            SELECT 
                p.patient_id,
                p.age,
                p.gender,
                COUNT(e.encounter_id) as total_encounters,
                MIN(e.admission_date) as first_admission,
                MAX(e.admission_date) as last_admission,
                AVG(JULIANDAY(e.discharge_date) - JULIANDAY(e.admission_date)) as avg_los
            FROM patients p
            LEFT JOIN encounters e ON p.patient_id = e.patient_id
            GROUP BY p.patient_id, p.age, p.gender
        ),
        encounter_rankings AS (
            SELECT 
                patient_id,
                total_encounters,
                NTILE(4) OVER (ORDER BY total_encounters) as encounter_quartile,
                PERCENT_RANK() OVER (ORDER BY total_encounters) as encounter_percentile
            FROM patient_encounters
        )
        SELECT 
            pe.*,
            er.encounter_quartile,
            ROUND(er.encounter_percentile * 100, 2) as encounter_percentile_rank
        FROM patient_encounters pe
        JOIN encounter_rankings er ON pe.patient_id = er.patient_id
        ORDER BY pe.total_encounters DESC
        """
        
        return self.execute_query(query)
    
    def readmission_trend_analysis(self) -> pd.DataFrame:
        """
        Analyze readmission trends using window functions
        """
        query = """
        WITH encounter_sequence AS (
            SELECT 
                e1.patient_id,
                e1.encounter_id as index_encounter,
                e1.discharge_date as index_discharge,
                e2.encounter_id as next_encounter,
                e2.admission_date as next_admission,
                JULIANDAY(e2.admission_date) - JULIANDAY(e1.discharge_date) as days_to_next,
                ROW_NUMBER() OVER (
                    PARTITION BY e1.encounter_id 
                    ORDER BY e2.admission_date
                ) as readmission_sequence
            FROM encounters e1
            LEFT JOIN encounters e2 
                ON e1.patient_id = e2.patient_id 
                AND e2.admission_date > e1.discharge_date
        ),
        readmission_flags AS (
            SELECT 
                *,
                CASE 
                    WHEN days_to_next <= 30 AND readmission_sequence = 1 THEN 1 
                    ELSE 0 
                END as readmitted_30d,
                CASE 
                    WHEN days_to_next <= 90 AND readmission_sequence = 1 THEN 1 
                    ELSE 0 
                END as readmitted_90d
            FROM encounter_sequence
            WHERE readmission_sequence = 1 OR readmission_sequence IS NULL
        )
        SELECT 
            strftime('%Y-%m', index_discharge) as discharge_month,
            COUNT(*) as total_discharges,
            SUM(readmitted_30d) as readmissions_30d,
            SUM(readmitted_90d) as readmissions_90d,
            ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2) as readmission_rate_30d,
            ROUND(100.0 * SUM(readmitted_90d) / COUNT(*), 2) as readmission_rate_90d,
            AVG(days_to_next) as avg_days_to_readmission
        FROM readmission_flags
        GROUP BY discharge_month
        ORDER BY discharge_month
        """
        
        return self.execute_query(query)
    
    def lab_value_trends(self) -> pd.DataFrame:
        """
        Analyze lab value trends with LEAD/LAG functions
        """
        query = """
        WITH ranked_labs AS (
            SELECT 
                patient_id,
                lab_name,
                lab_value,
                lab_date,
                LAG(lab_value) OVER (
                    PARTITION BY patient_id, lab_name 
                    ORDER BY lab_date
                ) as previous_value,
                LEAD(lab_value) OVER (
                    PARTITION BY patient_id, lab_name 
                    ORDER BY lab_date
                ) as next_value,
                ROW_NUMBER() OVER (
                    PARTITION BY patient_id, lab_name 
                    ORDER BY lab_date DESC
                ) as recency_rank
            FROM labs
        ),
        lab_changes AS (
            SELECT 
                *,
                lab_value - previous_value as change_from_previous,
                CASE 
                    WHEN previous_value IS NOT NULL THEN 
                        ROUND(100.0 * (lab_value - previous_value) / previous_value, 2)
                    ELSE NULL
                END as pct_change_from_previous
            FROM ranked_labs
        )
        SELECT 
            patient_id,
            lab_name,
            lab_value as current_value,
            previous_value,
            next_value,
            change_from_previous,
            pct_change_from_previous,
            recency_rank
        FROM lab_changes
        WHERE recency_rank <= 5
        ORDER BY patient_id, lab_name, lab_date DESC
        """
        
        return self.execute_query(query)
    
    def medication_adherence_analysis(self) -> pd.DataFrame:
        """
        Analyze medication adherence patterns
        """
        query = """
        WITH medication_timeline AS (
            SELECT 
                m.patient_id,
                m.medication_name,
                m.start_date,
                m.end_date,
                JULIANDAY(m.end_date) - JULIANDAY(m.start_date) as duration_days,
                COUNT(*) OVER (
                    PARTITION BY m.patient_id, m.medication_name
                ) as prescription_count,
                SUM(JULIANDAY(m.end_date) - JULIANDAY(m.start_date)) OVER (
                    PARTITION BY m.patient_id, m.medication_name
                ) as total_days_on_med
            FROM medications m
        ),
        adherence_metrics AS (
            SELECT 
                patient_id,
                medication_name,
                prescription_count,
                AVG(duration_days) as avg_prescription_duration,
                total_days_on_med,
                CASE 
                    WHEN total_days_on_med >= 270 THEN 'High Adherence'
                    WHEN total_days_on_med >= 180 THEN 'Moderate Adherence'
                    ELSE 'Low Adherence'
                END as adherence_category
            FROM medication_timeline
            GROUP BY patient_id, medication_name
        )
        SELECT 
            medication_name,
            adherence_category,
            COUNT(*) as patient_count,
            AVG(prescription_count) as avg_prescriptions,
            AVG(total_days_on_med) as avg_total_days
        FROM adherence_metrics
        GROUP BY medication_name, adherence_category
        ORDER BY medication_name, adherence_category
        """
        
        return self.execute_query(query)
    
    def diagnosis_co_occurrence(self) -> pd.DataFrame:
        """
        Analyze diagnosis co-occurrence patterns
        """
        query = """
        WITH patient_diagnoses AS (
            SELECT 
                d1.patient_id,
                d1.diagnosis_code as diagnosis_1,
                d2.diagnosis_code as diagnosis_2,
                COUNT(*) as co_occurrence_count
            FROM diagnoses d1
            JOIN diagnoses d2 
                ON d1.patient_id = d2.patient_id 
                AND d1.diagnosis_code < d2.diagnosis_code
            GROUP BY d1.patient_id, d1.diagnosis_code, d2.diagnosis_code
        ),
        diagnosis_pairs AS (
            SELECT 
                diagnosis_1,
                diagnosis_2,
                COUNT(DISTINCT patient_id) as patient_count,
                AVG(co_occurrence_count) as avg_co_occurrences
            FROM patient_diagnoses
            GROUP BY diagnosis_1, diagnosis_2
            HAVING patient_count >= 5
        )
        SELECT 
            diagnosis_1,
            diagnosis_2,
            patient_count,
            ROUND(avg_co_occurrences, 2) as avg_co_occurrences,
            RANK() OVER (ORDER BY patient_count DESC) as prevalence_rank
        FROM diagnosis_pairs
        ORDER BY patient_count DESC
        LIMIT 50
        """
        
        return self.execute_query(query)
    
    def length_of_stay_analysis(self) -> pd.DataFrame:
        """
        Comprehensive length of stay analysis with percentiles
        """
        query = """
        WITH los_calculations AS (
            SELECT 
                e.encounter_id,
                e.patient_id,
                p.age,
                p.gender,
                JULIANDAY(e.discharge_date) - JULIANDAY(e.admission_date) as los_days,
                COUNT(d.diagnosis_id) as diagnosis_count,
                COUNT(DISTINCT m.medication_name) as unique_medications
            FROM encounters e
            JOIN patients p ON e.patient_id = p.patient_id
            LEFT JOIN diagnoses d ON e.encounter_id = d.encounter_id
            LEFT JOIN medications m ON e.encounter_id = m.encounter_id
            GROUP BY e.encounter_id
        ),
        los_percentiles AS (
            SELECT 
                *,
                NTILE(10) OVER (ORDER BY los_days) as los_decile,
                PERCENT_RANK() OVER (ORDER BY los_days) as los_percentile
            FROM los_calculations
        )
        SELECT 
            los_decile,
            COUNT(*) as encounter_count,
            ROUND(AVG(los_days), 2) as avg_los,
            ROUND(MIN(los_days), 2) as min_los,
            ROUND(MAX(los_days), 2) as max_los,
            ROUND(AVG(age), 1) as avg_age,
            ROUND(AVG(diagnosis_count), 1) as avg_diagnoses,
            ROUND(AVG(unique_medications), 1) as avg_medications
        FROM los_percentiles
        GROUP BY los_decile
        ORDER BY los_decile
        """
        
        return self.execute_query(query)
    
    def high_risk_patient_identification(self) -> pd.DataFrame:
        """
        Identify high-risk patients using multiple criteria
        """
        query = """
        WITH patient_metrics AS (
            SELECT 
                p.patient_id,
                p.age,
                p.gender,
                COUNT(DISTINCT e.encounter_id) as encounter_count,
                COUNT(DISTINCT d.diagnosis_code) as unique_diagnoses,
                COUNT(DISTINCT m.medication_name) as unique_medications,
                AVG(JULIANDAY(e.discharge_date) - JULIANDAY(e.admission_date)) as avg_los,
                MAX(e.admission_date) as last_admission
            FROM patients p
            LEFT JOIN encounters e ON p.patient_id = e.patient_id
            LEFT JOIN diagnoses d ON p.patient_id = d.patient_id
            LEFT JOIN medications m ON p.patient_id = m.patient_id
            GROUP BY p.patient_id
        ),
        risk_scoring AS (
            SELECT 
                *,
                CASE WHEN age >= 75 THEN 2 WHEN age >= 65 THEN 1 ELSE 0 END +
                CASE WHEN encounter_count >= 5 THEN 2 WHEN encounter_count >= 3 THEN 1 ELSE 0 END +
                CASE WHEN unique_diagnoses >= 10 THEN 2 WHEN unique_diagnoses >= 5 THEN 1 ELSE 0 END +
                CASE WHEN avg_los >= 7 THEN 2 WHEN avg_los >= 4 THEN 1 ELSE 0 END as risk_score
            FROM patient_metrics
        )
        SELECT 
            patient_id,
            age,
            gender,
            encounter_count,
            unique_diagnoses,
            unique_medications,
            ROUND(avg_los, 1) as avg_los,
            risk_score,
            CASE 
                WHEN risk_score >= 6 THEN 'Very High Risk'
                WHEN risk_score >= 4 THEN 'High Risk'
                WHEN risk_score >= 2 THEN 'Moderate Risk'
                ELSE 'Low Risk'
            END as risk_category,
            last_admission
        FROM risk_scoring
        WHERE risk_score >= 4
        ORDER BY risk_score DESC, last_admission DESC
        """
        
        return self.execute_query(query)
    
    def seasonal_admission_patterns(self) -> pd.DataFrame:
        """
        Analyze seasonal patterns in admissions
        """
        query = """
        WITH admission_details AS (
            SELECT 
                strftime('%Y', admission_date) as year,
                strftime('%m', admission_date) as month,
                CASE 
                    WHEN CAST(strftime('%m', admission_date) AS INTEGER) IN (12, 1, 2) THEN 'Winter'
                    WHEN CAST(strftime('%m', admission_date) AS INTEGER) IN (3, 4, 5) THEN 'Spring'
                    WHEN CAST(strftime('%m', admission_date) AS INTEGER) IN (6, 7, 8) THEN 'Summer'
                    ELSE 'Fall'
                END as season,
                CASE 
                    WHEN CAST(strftime('%w', admission_date) AS INTEGER) IN (0, 6) THEN 'Weekend'
                    ELSE 'Weekday'
                END as day_type,
                encounter_id,
                patient_id
            FROM encounters
        ),
        seasonal_stats AS (
            SELECT 
                season,
                day_type,
                COUNT(*) as admission_count,
                COUNT(DISTINCT patient_id) as unique_patients,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct_of_total
            FROM admission_details
            GROUP BY season, day_type
        )
        SELECT 
            season,
            day_type,
            admission_count,
            unique_patients,
            pct_of_total,
            RANK() OVER (ORDER BY admission_count DESC) as volume_rank
        FROM seasonal_stats
        ORDER BY admission_count DESC
        """
        
        return self.execute_query(query)
    
    def get_query_templates(self) -> Dict[str, str]:
        """Get predefined query templates"""
        templates = {
            'patient_summary': """
                SELECT 
                    p.patient_id,
                    p.age,
                    p.gender,
                    COUNT(DISTINCT e.encounter_id) as total_encounters,
                    COUNT(DISTINCT d.diagnosis_code) as total_diagnoses
                FROM patients p
                LEFT JOIN encounters e ON p.patient_id = e.patient_id
                LEFT JOIN diagnoses d ON p.patient_id = d.patient_id
                WHERE p.patient_id = ?
                GROUP BY p.patient_id
            """,
            'recent_encounters': """
                SELECT 
                    e.encounter_id,
                    e.admission_date,
                    e.discharge_date,
                    JULIANDAY(e.discharge_date) - JULIANDAY(e.admission_date) as los
                FROM encounters e
                WHERE e.patient_id = ?
                ORDER BY e.admission_date DESC
                LIMIT 10
            """,
            'diagnosis_history': """
                SELECT 
                    d.diagnosis_code,
                    d.diagnosis_description,
                    COUNT(*) as occurrence_count,
                    MIN(d.diagnosis_date) as first_diagnosed,
                    MAX(d.diagnosis_date) as last_diagnosed
                FROM diagnoses d
                WHERE d.patient_id = ?
                GROUP BY d.diagnosis_code
                ORDER BY occurrence_count DESC
            """
        }
        
        return templates
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔒 Connection closed")


if __name__ == "__main__":
    # Example usage
    analytics = SQLAnalytics()
    analytics.connect()
    
    print("\n" + "="*60)
    print("COHORT ANALYSIS")
    print("="*60)
    cohort = analytics.patient_cohort_analysis()
    print(cohort.head())
    
    print("\n" + "="*60)
    print("READMISSION TRENDS")
    print("="*60)
    trends = analytics.readmission_trend_analysis()
    print(trends.head())
    
    print("\n" + "="*60)
    print("HIGH RISK PATIENTS")
    print("="*60)
    high_risk = analytics.high_risk_patient_identification()
    print(high_risk.head())
    
    analytics.close()
