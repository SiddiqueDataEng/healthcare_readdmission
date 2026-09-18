"""
Multi-Source Data Loader
Supports CSV, Excel, Parquet, Avro, and Database connections
"""

import pandas as pd
import numpy as np
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json
from typing import Union, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Universal data loader supporting multiple formats and sources
    """
    
    def __init__(self):
        self.data = None
        self.metadata = {}
        self.connection = None
        
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file"""
        print(f"📂 Loading CSV: {filepath}")
        self.data = pd.read_csv(filepath, **kwargs)
        self.metadata['source'] = filepath
        self.metadata['format'] = 'CSV'
        self.metadata['rows'] = len(self.data)
        self.metadata['columns'] = len(self.data.columns)
        print(f"✅ Loaded {self.metadata['rows']:,} rows × {self.metadata['columns']} columns")
        return self.data
    
    def load_excel(self, filepath: str, sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        """Load data from Excel file"""
        print(f"📊 Loading Excel: {filepath} (Sheet: {sheet_name})")
        self.data = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        self.metadata['source'] = filepath
        self.metadata['format'] = 'Excel'
        self.metadata['sheet'] = sheet_name
        self.metadata['rows'] = len(self.data)
        self.metadata['columns'] = len(self.data.columns)
        print(f"✅ Loaded {self.metadata['rows']:,} rows × {self.metadata['columns']} columns")
        return self.data
    
    def load_parquet(self, filepath: str) -> pd.DataFrame:
        """Load data from Parquet file"""
        print(f"🗄️ Loading Parquet: {filepath}")
        self.data = pd.read_parquet(filepath)
        self.metadata['source'] = filepath
        self.metadata['format'] = 'Parquet'
        self.metadata['rows'] = len(self.data)
        self.metadata['columns'] = len(self.data.columns)
        print(f"✅ Loaded {self.metadata['rows']:,} rows × {self.metadata['columns']} columns")
        return self.data
    
    def load_from_database(self, db_path: str, query: str = None, table_name: str = None) -> pd.DataFrame:
        """Load data from SQLite database"""
        print(f"🗃️ Connecting to database: {db_path}")
        self.connection = sqlite3.connect(db_path)
        
        if query:
            print(f"📝 Executing custom query...")
            self.data = pd.read_sql_query(query, self.connection)
        elif table_name:
            print(f"📋 Loading table: {table_name}")
            self.data = pd.read_sql_query(f"SELECT * FROM {table_name}", self.connection)
        else:
            raise ValueError("Either 'query' or 'table_name' must be provided")
        
        self.metadata['source'] = db_path
        self.metadata['format'] = 'SQLite'
        self.metadata['rows'] = len(self.data)
        self.metadata['columns'] = len(self.data.columns)
        print(f"✅ Loaded {self.metadata['rows']:,} rows × {self.metadata['columns']} columns")
        return self.data
    
    def load_multiple_csv(self, directory: str, pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
        """Load multiple CSV files from directory"""
        print(f"📁 Loading multiple CSV files from: {directory}")
        path = Path(directory)
        files = list(path.glob(pattern))
        
        datasets = {}
        for file in files:
            name = file.stem
            print(f"  Loading {name}...")
            datasets[name] = pd.read_csv(file)
            print(f"    ✅ {len(datasets[name]):,} rows")
        
        print(f"✅ Loaded {len(datasets)} datasets")
        return datasets
    
    def save_to_parquet(self, filepath: str, compression: str = 'snappy'):
        """Save data to Parquet format"""
        if self.data is None:
            raise ValueError("No data loaded to save")
        
        print(f"💾 Saving to Parquet: {filepath}")
        self.data.to_parquet(filepath, compression=compression, index=False)
        
        # Get file size
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        print(f"✅ Saved {len(self.data):,} rows ({size_mb:.2f} MB)")
    
    def save_to_avro(self, filepath: str):
        """Save data to Avro format (via Parquet with Avro schema)"""
        if self.data is None:
            raise ValueError("No data loaded to save")
        
        print(f"💾 Saving to Avro-compatible format: {filepath}")
        # Convert to PyArrow table and save
        table = pa.Table.from_pandas(self.data)
        pq.write_table(table, filepath)
        
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        print(f"✅ Saved {len(self.data):,} rows ({size_mb:.2f} MB)")
    
    def describe_data(self) -> Dict:
        """Get comprehensive data description"""
        if self.data is None:
            return {}
        
        # Convert nullable integer types to regular types for JSON serialization
        data_copy = self.data.copy()
        for col in data_copy.columns:
            if pd.api.types.is_integer_dtype(data_copy[col]):
                data_copy[col] = data_copy[col].astype('Int64').astype(float)
        
        description = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': {k: str(v) for k, v in self.data.dtypes.to_dict().items()},
            'missing_values': {k: int(v) for k, v in self.data.isnull().sum().to_dict().items()},
            'missing_percentage': {k: float(v) for k, v in (self.data.isnull().sum() / len(self.data) * 100).to_dict().items()},
            'memory_usage_mb': float(self.data.memory_usage(deep=True).sum() / (1024 * 1024)),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(self.data.select_dtypes(include=['datetime64']).columns)
        }
        
        return description
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """Get sample of data"""
        if self.data is None:
            return pd.DataFrame()
        return self.data.head(n)
    
    def close_connection(self):
        """Close database connection if open"""
        if self.connection:
            self.connection.close()
            print("🔒 Database connection closed")


class DataCleaner:
    """
    Comprehensive data cleaning and wrangling utilities
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.original_shape = data.shape
        self.cleaning_log = []
        
    def handle_missing_values(self, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values with various strategies
        
        Strategies:
        - 'drop': Remove rows with missing values
        - 'mean': Fill with mean (numeric only)
        - 'median': Fill with median (numeric only)
        - 'mode': Fill with mode
        - 'forward_fill': Forward fill
        - 'backward_fill': Backward fill
        - 'constant': Fill with constant value
        """
        print("🧹 Handling missing values...")
        
        missing_before = self.data.isnull().sum().sum()
        
        if strategy is None:
            # Default strategy
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
            
            # Fill numeric with median
            for col in numeric_cols:
                if self.data[col].isnull().any():
                    median_val = self.data[col].median()
                    self.data[col].fillna(median_val, inplace=True)
                    self.cleaning_log.append(f"Filled {col} with median: {median_val:.2f}")
            
            # Fill categorical with mode
            for col in categorical_cols:
                if self.data[col].isnull().any():
                    mode_val = self.data[col].mode()[0] if not self.data[col].mode().empty else 'Unknown'
                    self.data[col].fillna(mode_val, inplace=True)
                    self.cleaning_log.append(f"Filled {col} with mode: {mode_val}")
        else:
            # Custom strategy per column
            for col, method in strategy.items():
                if col not in self.data.columns:
                    continue
                
                if method == 'drop':
                    self.data.dropna(subset=[col], inplace=True)
                elif method == 'mean':
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif method == 'median':
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif method == 'mode':
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                elif method == 'forward_fill':
                    self.data[col].fillna(method='ffill', inplace=True)
                elif method == 'backward_fill':
                    self.data[col].fillna(method='bfill', inplace=True)
                
                self.cleaning_log.append(f"Applied {method} to {col}")
        
        missing_after = self.data.isnull().sum().sum()
        print(f"  ✅ Reduced missing values: {missing_before:,} → {missing_after:,}")
        
        return self.data
    
    def remove_duplicates(self, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows"""
        print("🔍 Removing duplicates...")
        
        duplicates_before = self.data.duplicated(subset=subset).sum()
        self.data.drop_duplicates(subset=subset, keep=keep, inplace=True)
        duplicates_removed = duplicates_before
        
        print(f"  ✅ Removed {duplicates_removed:,} duplicate rows")
        self.cleaning_log.append(f"Removed {duplicates_removed} duplicates")
        
        return self.data
    
    def handle_outliers(self, columns: List[str] = None, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers using IQR or Z-score method
        
        Methods:
        - 'iqr': Interquartile range (default threshold=1.5)
        - 'zscore': Z-score (default threshold=3)
        """
        print(f"📊 Handling outliers using {method} method...")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        outliers_removed = 0
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            before_count = len(self.data)
            
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                self.data = self.data[(z_scores < threshold) | self.data[col].isnull()]
            
            after_count = len(self.data)
            removed = before_count - after_count
            outliers_removed += removed
            
            if removed > 0:
                self.cleaning_log.append(f"Removed {removed} outliers from {col}")
        
        print(f"  ✅ Removed {outliers_removed:,} outlier rows")
        
        return self.data
    
    def convert_datatypes(self, conversions: Dict[str, str]) -> pd.DataFrame:
        """Convert column datatypes"""
        print("🔄 Converting datatypes...")
        
        for col, dtype in conversions.items():
            if col not in self.data.columns:
                continue
            
            try:
                if dtype == 'datetime':
                    self.data[col] = pd.to_datetime(self.data[col])
                elif dtype == 'category':
                    self.data[col] = self.data[col].astype('category')
                else:
                    self.data[col] = self.data[col].astype(dtype)
                
                self.cleaning_log.append(f"Converted {col} to {dtype}")
                print(f"  ✅ {col} → {dtype}")
            except Exception as e:
                print(f"  ⚠️ Failed to convert {col}: {str(e)}")
        
        return self.data
    
    def normalize_text(self, columns: List[str], lowercase: bool = True, strip: bool = True) -> pd.DataFrame:
        """Normalize text columns"""
        print("📝 Normalizing text...")
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            if lowercase:
                self.data[col] = self.data[col].str.lower()
            if strip:
                self.data[col] = self.data[col].str.strip()
            
            self.cleaning_log.append(f"Normalized text in {col}")
            print(f"  ✅ {col}")
        
        return self.data
    
    def create_derived_features(self, feature_definitions: Dict[str, callable]) -> pd.DataFrame:
        """Create derived features using custom functions"""
        print("⚙️ Creating derived features...")
        
        for feature_name, func in feature_definitions.items():
            try:
                self.data[feature_name] = func(self.data)
                self.cleaning_log.append(f"Created feature: {feature_name}")
                print(f"  ✅ {feature_name}")
            except Exception as e:
                print(f"  ⚠️ Failed to create {feature_name}: {str(e)}")
        
        return self.data
    
    def get_cleaning_report(self) -> Dict:
        """Get comprehensive cleaning report"""
        report = {
            'original_shape': self.original_shape,
            'final_shape': self.data.shape,
            'rows_removed': self.original_shape[0] - self.data.shape[0],
            'columns_added': self.data.shape[1] - self.original_shape[1],
            'cleaning_steps': len(self.cleaning_log),
            'cleaning_log': self.cleaning_log,
            'data_quality': {
                'missing_values': self.data.isnull().sum().sum(),
                'duplicate_rows': self.data.duplicated().sum(),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024 * 1024)
            }
        }
        
        return report
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return cleaned data"""
        return self.data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load from CSV
    data = loader.load_csv('data/raw/patients.csv')
    print("\n" + "="*60)
    print("DATA DESCRIPTION")
    print("="*60)
    desc = loader.describe_data()
    print(f"Shape: {desc['shape']}")
    print(f"Memory: {desc['memory_usage_mb']:.2f} MB")
    print(f"Numeric columns: {len(desc['numeric_columns'])}")
    print(f"Categorical columns: {len(desc['categorical_columns'])}")
    
    # Clean data
    cleaner = DataCleaner(data)
    cleaned = cleaner.handle_missing_values()
    cleaned = cleaner.remove_duplicates()
    
    # Get report
    report = cleaner.get_cleaning_report()
    print("\n" + "="*60)
    print("CLEANING REPORT")
    print("="*60)
    print(f"Original shape: {report['original_shape']}")
    print(f"Final shape: {report['final_shape']}")
    print(f"Rows removed: {report['rows_removed']}")
    print(f"Cleaning steps: {report['cleaning_steps']}")
    
    # Save to Parquet
    loader.data = cleaned
    loader.save_to_parquet('data/processed/patients_cleaned.parquet')
