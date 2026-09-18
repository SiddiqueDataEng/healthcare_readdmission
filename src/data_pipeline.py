# src/data_pipeline.py
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPipeline:
    def __init__(self, db_path='data/raw/ehr_synthetic.db'):
        self.db_path = db_path
        self.conn = None
        self.raw_data = None
        self.processed_data = None
        
    def connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to {self.db_path}")
        
    def extract_data(self, sql_file='sql/sql01_extract_index_admissions.sql'):
        """Execute SQL extraction script"""
        with open(sql_file, 'r') as f:
            sql_query = f.read()
        
        self.raw_data = pd.read_sql_query(sql_query, self.conn)
        print(f"Extracted {len(self.raw_data)} index admissions")
        print(f"Readmission rate: {self.raw_data['readmitted_30d'].mean():.2%}")
        
        return self.raw_data
    
    def clean_data(self):
        """Clean and preprocess the data"""
        df = self.raw_data.copy()
        
        # Handle missing values
        print("\nMissing values before cleaning:")
        print(df.isnull().sum())
        
        # Impute missing lab values with median
        lab_columns = ['last_bnp', 'last_sodium', 'last_creatinine', 'last_hemoglobin']
        for col in lab_columns:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Imputed {col} with median: {median_val:.2f}")
        
        # Convert categorical variables
        categorical_cols = ['gender', 'race', 'insurance_type', 'discharge_disposition',
                           'discharge_location', 'insurance_category']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Create derived features
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 65, 75, 85, 120],
                                 labels=['<65', '65-74', '75-84', '85+'])
        
        df['bnp_category'] = pd.cut(df['last_bnp'],
                                   bins=[0, 100, 400, 1000, float('inf')],
                                   labels=['Normal', 'Mild', 'Moderate', 'Severe'])
        
        # Comorbidity score (simple sum)
        comorbidity_cols = ['has_hypertension', 'has_diabetes', 'has_cad', 
                           'has_copd', 'has_ckd', 'has_afib', 'has_obesity', 'has_anemia']
        df['comorbidity_score'] = df[comorbidity_cols].sum(axis=1)
        
        self.processed_data = df
        print(f"\nData cleaned. Shape: {df.shape}")
        
        return df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        df = self.processed_data
        
        # Create EDA directory
        import os
        os.makedirs('reports/eda', exist_ok=True)
        
        # 1. Target variable distribution
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='readmitted_30d', data=df)
        plt.title('Distribution of 30-Day Readmissions')
        plt.xlabel('Readmitted within 30 Days')
        plt.ylabel('Count')
        
        # Add percentages on bars
        total = len(df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height()/total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height() + 0.5
            ax.annotate(percentage, (x, y), ha='center')
        
        plt.savefig('reports/eda/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Numerical features vs readmission
        numerical_features = ['age', 'last_bnp', 'last_creatinine', 'length_of_stay', 
                             'comorbidity_score', 'total_diagnoses']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(numerical_features[:6]):
            sns.boxplot(x='readmitted_30d', y=feature, data=df, ax=axes[i])
            axes[i].set_title(f'{feature} vs Readmission')
            axes[i].set_xlabel('Readmitted')
            axes[i].set_ylabel(feature)
        
        plt.tight_layout()
        plt.savefig('reports/eda/numerical_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Categorical features vs readmission
        categorical_features = ['gender', 'race', 'discharge_location', 'age_group', 'bnp_category']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(categorical_features[:6]):
            if feature in df.columns:
                # Calculate readmission rates by category
                rate_df = df.groupby(feature)['readmitted_30d'].mean().reset_index()
                rate_df = rate_df.sort_values('readmitted_30d', ascending=False)
                
                ax = axes[i]
                sns.barplot(x=feature, y='readmitted_30d', data=rate_df, ax=ax)
                ax.set_title(f'Readmission Rate by {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Readmission Rate')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/eda/categorical_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.savefig('reports/eda/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Create summary statistics table
        summary_stats = pd.DataFrame({
            'Feature': [],
            'Readmitted_Mean': [],
            'Not_Readmitted_Mean': [],
            'P_Value': []
        })
        
        from scipy.stats import ttest_ind
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'readmitted_30d':
                readmitted = df[df['readmitted_30d'] == 1][col]
                not_readmitted = df[df['readmitted_30d'] == 0][col]
                
                if len(readmitted) > 1 and len(not_readmitted) > 1:
                    t_stat, p_val = ttest_ind(readmitted, not_readmitted, nan_policy='omit')
                    
                    summary_stats = pd.concat([summary_stats, pd.DataFrame({
                        'Feature': [col],
                        'Readmitted_Mean': [readmitted.mean()],
                        'Not_Readmitted_Mean': [not_readmitted.mean()],
                        'P_Value': [p_val]
                    })])
        
        summary_stats['Significant'] = summary_stats['P_Value'] < 0.05
        summary_stats.to_csv('reports/eda/univariate_analysis.csv', index=False)
        
        print("Exploratory analysis complete. Check 'reports/eda/' directory.")
        
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """Create train/test splits"""
        df = self.processed_data
        
        # Features and target
        X = df.drop(['readmitted_30d', 'index_encounter_id', 'patient_id', 
                     'index_admission', 'index_discharge', 'readmission_date',
                     'days_to_readmit', 'died_within_30d'], axis=1, errors='ignore')
        y = df['readmitted_30d']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.1%})")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.1%})")
        print(f"Train readmission rate: {y_train.mean():.2%}")
        print(f"Test readmission rate: {y_test.mean():.2%}")
        
        # Save the splits
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv('data/processed/train_data.csv', index=False)
        test_data.to_csv('data/processed/test_data.csv', index=False)
        
        return X_train, X_test, y_train, y_test
    
    def generate_data_dictionary(self):
        """Create a data dictionary for documentation"""
        df = self.processed_data
        
        data_dict = []
        for col in df.columns:
            col_dict = {
                'Variable_Name': col,
                'Description': self._get_variable_description(col),
                'Data_Type': str(df[col].dtype),
                'Missing_Values': df[col].isnull().sum(),
                'Missing_Percentage': f"{df[col].isnull().mean():.1%}",
                'Unique_Values': df[col].nunique(),
                'Sample_Values': str(df[col].dropna().unique()[:3].tolist() if df[col].nunique() > 3 else df[col].dropna().unique().tolist())
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_dict.update({
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Mean': df[col].mean(),
                    'Median': df[col].median()
                })
            
            data_dict.append(col_dict)
        
        data_dict_df = pd.DataFrame(data_dict)
        data_dict_df.to_csv('docs/protocols/data_dictionary.csv', index=False)
        print("Data dictionary saved to 'docs/protocols/data_dictionary.csv'")
        
        return data_dict_df
    
    def _get_variable_description(self, col_name):
        """Helper function to get variable descriptions"""
        descriptions = {
            'readmitted_30d': 'Binary indicator: 1 if patient readmitted within 30 days of discharge',
            'age': 'Patient age in years',
            'gender': 'Patient gender',
            'last_bnp': 'Most recent B-type Natriuretic Peptide lab value',
            'last_creatinine': 'Most recent creatinine lab value',
            'length_of_stay': 'Days between admission and discharge',
            'comorbidity_score': 'Count of comorbid conditions',
            'has_hypertension': 'Binary: 1 if hypertension diagnosis present',
            'has_diabetes': 'Binary: 1 if diabetes diagnosis present',
            'weekend_discharge': 'Binary: 1 if discharged on weekend',
            'discharge_location': 'Category: Home or Facility discharge',
            'hf_med_count': 'Count of heart failure medications prescribed'
        }
        
        return descriptions.get(col_name, 'No description available')
    
    def run_full_pipeline(self):
        """Run the complete data pipeline"""
        print("="*60)
        print("STARTING DATA PREPARATION PIPELINE")
        print("="*60)
        
        # Step 1: Connect to database
        self.connect()
        
        # Step 2: Extract data
        print("\n1. Extracting data from database...")
        self.extract_data()
        
        # Step 3: Clean data
        print("\n2. Cleaning and preprocessing data...")
        self.clean_data()
        
        # Step 4: Generate data dictionary
        print("\n3. Generating data dictionary...")
        self.generate_data_dictionary()
        
        # Step 5: Exploratory analysis
        print("\n4. Performing exploratory analysis...")
        self.exploratory_analysis()
        
        # Step 6: Create train/test split
        print("\n5. Creating train/test splits...")
        X_train, X_test, y_train, y_test = self.create_train_test_split()
        
        # Step 7: Save processed data
        self.processed_data.to_csv('data/processed/full_dataset.csv', index=False)
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60)
        print(f"\nFinal dataset: {self.processed_data.shape[0]} patients")
        print(f"Readmission rate: {self.processed_data['readmitted_30d'].mean():.2%}")
        print(f"Files saved to 'data/processed/'")
        print(f"Reports saved to 'reports/eda/'")
        
        return X_train, X_test, y_train, y_test

# Main execution
if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_full_pipeline()