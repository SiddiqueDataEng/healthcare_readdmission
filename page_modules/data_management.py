"""
Data Management & Cleaning Module
Load, clean, wrangle, and export data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import DataLoader, DataCleaner


def convert_to_serializable(df):
    """Convert DataFrame to JSON-serializable types"""
    df_copy = df.copy()
    for col in df_copy.columns:
        # Convert Int64, Int32, etc. to float
        if pd.api.types.is_integer_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(float)
        # Convert other nullable types
        elif hasattr(df_copy[col].dtype, 'numpy_dtype'):
            df_copy[col] = df_copy[col].astype(float)
    return df_copy


def show():
    """Data Management page"""
    
    st.title("📦 Data Management & Cleaning")
    st.markdown("### Load, explore, clean, and export your data")
    
    # Add helpful info box
    with st.expander("ℹ️ How to Get Started", expanded=False):
        st.markdown("""
        **🚀 Quick Start Options:**
        1. **Generate Sample Data** (Easiest): Click "Generate Sample Data" to create demo healthcare data instantly
        2. **Use Existing Data**: Click "Load Full Dataset" if you've already run the setup scripts
        3. **Upload Your Own**: Use the file upload options below to load CSV, Excel, or other formats
        4. **Connect to Database**: Load data directly from SQLite databases
        
        **💡 Tips:**
        - Start with "Generate Sample Data" if you're new to the platform
        - The sample data includes 100 patients with realistic healthcare attributes
        - You can always load your own data later using the upload options below
        """)
    
    st.markdown("---")
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    loader = st.session_state.data_loader
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂 Load Data",
        "🔍 Explore Data",
        "🧹 Clean Data",
        "💾 Export Data",
        "📊 Data Quality Report"
    ])
    
    # ==================== TAB 1: Load Data ====================
    with tab1:
        st.subheader("Load Data from Multiple Sources")
        
        # Load existing processed data
        st.markdown("#### Quick Start: Load Existing Data")
        st.info("💡 **Recommended:** Start with existing processed data, then try loading your own files below.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Load Full Dataset", type="primary", use_container_width=True):
                try:
                    data = pd.read_csv('data/processed/full_dataset.csv')
                    st.session_state.loaded_data = data
                    loader.data = data
                    st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                    st.dataframe(data.head(), use_container_width=True)
                except FileNotFoundError:
                    st.error("❌ File not found. Try 'Generate Sample Data' button.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🎯 Load Train Data", use_container_width=True):
                try:
                    data = pd.read_csv('data/processed/train_data.csv')
                    st.session_state.loaded_data = data
                    loader.data = data
                    st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                    st.dataframe(data.head(), use_container_width=True)
                except FileNotFoundError:
                    st.error("❌ File not found. Try 'Generate Sample Data' button.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col3:
            if st.button("🧪 Load Test Data", use_container_width=True):
                try:
                    data = pd.read_csv('data/processed/test_data.csv')
                    st.session_state.loaded_data = data
                    loader.data = data
                    st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                    st.dataframe(data.head(), use_container_width=True)
                except FileNotFoundError:
                    st.error("❌ File not found. Try 'Generate Sample Data' button.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col4:
            if st.button("🎲 Generate Sample Data", use_container_width=True):
                try:
                    with st.spinner("Generating sample data..."):
                        # Import and run the quick data generator
                        import sys
                        sys.path.append(str(Path(__file__).parent.parent))
                        from src.quick_data_generator import generate_sample_data
                        
                        df = generate_sample_data()
                        st.session_state.loaded_data = df
                        loader.data = df
                        
                    st.success(f"✅ Generated and loaded {len(df):,} rows of sample data!")
                    st.info("💡 Sample data includes patients with demographics, conditions, and readmission flags.")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
        
        st.markdown("---")
        st.markdown("#### Or Load Your Own Data")
        
        source_type = st.radio(
            "Select Data Source",
            ["CSV File", "Excel File", "Parquet File", "Database Query", "Multiple CSV Files"]
        )
        
        if source_type == "CSV File":
            st.markdown("#### Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            col1, col2 = st.columns(2)
            with col1:
                delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"])
            with col2:
                encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1"])
            
            if uploaded_file is not None:
                if st.button("📥 Load CSV", type="primary"):
                    try:
                        with st.spinner("Loading CSV..."):
                            data = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding)
                            st.session_state.loaded_data = data
                            loader.data = data
                        
                        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                        st.dataframe(data.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error loading CSV: {str(e)}")
        
        elif source_type == "Excel File":
            st.markdown("#### Upload Excel File")
            uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
            
            if uploaded_file is not None:
                # Get sheet names
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("Select Sheet", excel_file.sheet_names)
                
                if st.button("📥 Load Excel", type="primary"):
                    try:
                        with st.spinner("Loading Excel..."):
                            data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                            st.session_state.loaded_data = data
                            loader.data = data
                        
                        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                        st.dataframe(data.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error loading Excel: {str(e)}")
        
        elif source_type == "Parquet File":
            st.markdown("#### Upload Parquet File")
            uploaded_file = st.file_uploader("Choose a Parquet file", type=['parquet'])
            
            if uploaded_file is not None:
                if st.button("📥 Load Parquet", type="primary"):
                    try:
                        with st.spinner("Loading Parquet..."):
                            data = pd.read_parquet(uploaded_file)
                            st.session_state.loaded_data = data
                            loader.data = data
                        
                        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                        st.dataframe(data.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error loading Parquet: {str(e)}")
        
        elif source_type == "Database Query":
            st.markdown("#### Load from Database")
            
            db_path = st.text_input("Database Path", value="data/raw/ehr_synthetic.db")
            
            query_option = st.radio("Query Type", ["Select Table", "Custom Query"])
            
            if query_option == "Select Table":
                table_name = st.text_input("Table Name", value="patients")
                
                if st.button("📥 Load Table", type="primary"):
                    try:
                        with st.spinner(f"Loading table {table_name}..."):
                            data = loader.load_from_database(db_path, table_name=table_name)
                            st.session_state.loaded_data = data
                        
                        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                        st.dataframe(data.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error loading table: {str(e)}")
            
            else:
                query = st.text_area("SQL Query", height=150, 
                                    placeholder="SELECT * FROM patients LIMIT 100")
                
                if st.button("📥 Execute Query", type="primary"):
                    try:
                        with st.spinner("Executing query..."):
                            data = loader.load_from_database(db_path, query=query)
                            st.session_state.loaded_data = data
                        
                        st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns)} columns")
                        st.dataframe(data.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error executing query: {str(e)}")
        
        elif source_type == "Multiple CSV Files":
            st.markdown("#### Load Multiple CSV Files from Directory")
            st.info("This will load all CSV files from the data/raw directory")
            
            if st.button("📥 Load All CSV Files", type="primary"):
                try:
                    with st.spinner("Loading multiple CSV files..."):
                        datasets = loader.load_multiple_csv('data/raw')
                    
                    st.success(f"✅ Loaded {len(datasets)} datasets")
                    
                    for name, data in datasets.items():
                        with st.expander(f"📄 {name} ({len(data):,} rows)"):
                            st.dataframe(data.head(), use_container_width=True)
                    
                    # Optionally merge datasets
                    if st.checkbox("Merge datasets?"):
                        merge_key = st.text_input("Merge key (column name)", value="patient_id")
                        if st.button("Merge"):
                            merged = None
                            for name, data in datasets.items():
                                if merge_key in data.columns:
                                    if merged is None:
                                        merged = data
                                    else:
                                        merged = merged.merge(data, on=merge_key, how='outer')
                            
                            if merged is not None:
                                st.session_state.loaded_data = merged
                                st.success(f"✅ Merged into {len(merged):,} rows × {len(merged.columns)} columns")
                                st.dataframe(merged.head(), use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error loading files: {str(e)}")
    
    # ==================== TAB 2: Explore Data ====================
    with tab2:
        if st.session_state.loaded_data is None:
            st.warning("⚠️ Please load data first in the 'Load Data' tab")
            
            # Offer to load existing processed data
            st.markdown("---")
            st.markdown("#### Or Load Existing Processed Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Load Full Dataset", type="primary"):
                    try:
                        data = pd.read_csv('data/processed/full_dataset.csv')
                        st.session_state.loaded_data = data
                        st.success(f"✅ Loaded {len(data):,} rows")
                        st.rerun()
                    except FileNotFoundError:
                        st.error("❌ File not found. Run START_HERE.bat first.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("🎯 Load Train Data"):
                    try:
                        data = pd.read_csv('data/processed/train_data.csv')
                        st.session_state.loaded_data = data
                        st.success(f"✅ Loaded {len(data):,} rows")
                        st.rerun()
                    except FileNotFoundError:
                        st.error("❌ File not found. Run START_HERE.bat first.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            with col3:
                if st.button("🧪 Load Test Data"):
                    try:
                        data = pd.read_csv('data/processed/test_data.csv')
                        st.session_state.loaded_data = data
                        st.success(f"✅ Loaded {len(data):,} rows")
                        st.rerun()
                    except FileNotFoundError:
                        st.error("❌ File not found. Run START_HERE.bat first.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            st.markdown("---")
            st.info("💡 **Tip:** If files are missing, run `START_HERE.bat` to generate the data first.")
            
        else:
            data = st.session_state.loaded_data
            
            st.subheader("Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(data):,}")
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                st.metric("Memory", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            with col4:
                st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
            
            # Data preview
            st.markdown("#### Data Preview")
            n_rows = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(data.head(n_rows), use_container_width=True)
            
            # Column information
            st.markdown("#### Column Information")
            
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes.values,
                'Non-Null': data.count().values,
                'Null': data.isnull().sum().values,
                'Null %': (data.isnull().sum() / len(data) * 100).values,
                'Unique': data.nunique().values
            })
            
            st.dataframe(col_info, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### Statistical Summary")
            st.dataframe(data.describe(), use_container_width=True)
            
            # Missing values visualization
            st.markdown("#### Missing Values Analysis")
            
            missing_data = data.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                # Convert to DataFrame for plotting
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing_Count': missing_data.values.astype(float)
                })
                
                fig = px.bar(missing_df, x='Column', y='Missing_Count',
                           labels={'Missing_Count': 'Missing Count'},
                           title='Missing Values by Column')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Data types distribution
            st.markdown("#### Data Types Distribution")
            
            dtype_counts = data.dtypes.value_counts()
            dtype_df = pd.DataFrame({
                'Type': dtype_counts.index.astype(str),
                'Count': dtype_counts.values.astype(float)
            })
            
            fig = px.pie(dtype_df, values='Count', names='Type',
                        title='Column Data Types')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: Clean Data ====================
    with tab3:
        if st.session_state.loaded_data is None:
            st.warning("⚠️ Please load data first in the 'Load Data' tab")
            
            # Offer to load existing processed data
            st.markdown("#### Quick Load Options")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Load Full Dataset for Cleaning", type="primary"):
                    try:
                        data = pd.read_csv('data/processed/full_dataset.csv')
                        st.session_state.loaded_data = data
                        st.success(f"✅ Loaded {len(data):,} rows")
                        st.rerun()
                    except FileNotFoundError:
                        st.error("❌ File not found. Run START_HERE.bat first.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("📂 Go to Load Data Tab"):
                    st.info("👆 Click the 'Load Data' tab above to load your data first.")
            
        else:
            data = st.session_state.loaded_data
            
            st.subheader("Data Cleaning Operations")
            
            # Initialize cleaner
            cleaner = DataCleaner(data)
            
            # Missing values
            with st.expander("🔹 Handle Missing Values", expanded=True):
                st.markdown("**Current missing values:**")
                missing = data.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    st.dataframe(missing, use_container_width=True)
                else:
                    st.success("No missing values!")
                
                strategy = st.radio(
                    "Select Strategy",
                    ["Auto (Median for numeric, Mode for categorical)",
                     "Drop rows with missing values",
                     "Forward fill",
                     "Backward fill"]
                )
                
                if st.button("Apply Missing Value Strategy"):
                    with st.spinner("Handling missing values..."):
                        if strategy == "Auto (Median for numeric, Mode for categorical)":
                            cleaner.handle_missing_values()
                        elif strategy == "Drop rows with missing values":
                            cleaner.data.dropna(inplace=True)
                        elif strategy == "Forward fill":
                            cleaner.data.fillna(method='ffill', inplace=True)
                        elif strategy == "Backward fill":
                            cleaner.data.fillna(method='bfill', inplace=True)
                    
                    st.success("✅ Missing values handled!")
                    st.session_state.cleaned_data = cleaner.data
            
            # Duplicates
            with st.expander("🔹 Remove Duplicates"):
                duplicates = data.duplicated().sum()
                st.metric("Duplicate Rows", f"{duplicates:,}")
                
                if duplicates > 0:
                    if st.button("Remove Duplicates"):
                        with st.spinner("Removing duplicates..."):
                            cleaner.remove_duplicates()
                        st.success(f"✅ Removed {duplicates:,} duplicate rows!")
                        st.session_state.cleaned_data = cleaner.data
                else:
                    st.success("No duplicates found!")
            
            # Outliers
            with st.expander("🔹 Handle Outliers"):
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    selected_cols = st.multiselect("Select columns to check for outliers", numeric_cols)
                    
                    method = st.radio("Method", ["IQR (Interquartile Range)", "Z-Score"])
                    threshold = st.slider("Threshold", 1.0, 5.0, 1.5 if method == "IQR (Interquartile Range)" else 3.0)
                    
                    if selected_cols and st.button("Remove Outliers"):
                        with st.spinner("Removing outliers..."):
                            method_name = 'iqr' if 'IQR' in method else 'zscore'
                            cleaner.handle_outliers(columns=selected_cols, method=method_name, threshold=threshold)
                        st.success("✅ Outliers removed!")
                        st.session_state.cleaned_data = cleaner.data
                else:
                    st.info("No numeric columns found")
            
            # Data type conversion
            with st.expander("🔹 Convert Data Types"):
                st.markdown("**Current data types:**")
                st.dataframe(pd.DataFrame({'Column': data.columns, 'Type': data.dtypes.values}),
                           use_container_width=True)
                
                col_to_convert = st.selectbox("Select column", data.columns)
                new_type = st.selectbox("New type", ["int", "float", "str", "category", "datetime"])
                
                if st.button("Convert"):
                    try:
                        cleaner.convert_datatypes({col_to_convert: new_type})
                        st.success(f"✅ Converted {col_to_convert} to {new_type}")
                        st.session_state.cleaned_data = cleaner.data
                    except Exception as e:
                        st.error(f"❌ Conversion failed: {str(e)}")
            
            # Text normalization
            with st.expander("🔹 Normalize Text"):
                text_cols = data.select_dtypes(include=['object']).columns.tolist()
                
                if text_cols:
                    selected_text_cols = st.multiselect("Select text columns", text_cols)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        lowercase = st.checkbox("Convert to lowercase", value=True)
                    with col2:
                        strip = st.checkbox("Strip whitespace", value=True)
                    
                    if selected_text_cols and st.button("Normalize Text"):
                        with st.spinner("Normalizing text..."):
                            cleaner.normalize_text(selected_text_cols, lowercase=lowercase, strip=strip)
                        st.success("✅ Text normalized!")
                        st.session_state.cleaned_data = cleaner.data
                else:
                    st.info("No text columns found")
            
            # Get cleaning report
            st.markdown("---")
            if st.button("📊 Generate Cleaning Report"):
                report = cleaner.get_cleaning_report()
                
                st.subheader("Cleaning Report")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Rows", f"{report['original_shape'][0]:,}")
                    st.metric("Final Rows", f"{report['final_shape'][0]:,}")
                with col2:
                    st.metric("Rows Removed", f"{report['rows_removed']:,}")
                    st.metric("Columns Added", report['columns_added'])
                with col3:
                    st.metric("Cleaning Steps", report['cleaning_steps'])
                    st.metric("Final Memory", f"{report['data_quality']['memory_usage_mb']:.2f} MB")
                
                st.markdown("**Cleaning Log:**")
                for step in report['cleaning_log']:
                    st.text(f"• {step}")
                
                st.session_state.cleaned_data = cleaner.get_cleaned_data()
    
    # ==================== TAB 4: Export Data ====================
    with tab4:
        data_to_export = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.loaded_data
        
        if data_to_export is None:
            st.warning("⚠️ No data to export. Please load and optionally clean data first.")
        else:
            st.subheader("Export Data")
            
            st.info(f"📊 Ready to export: {len(data_to_export):,} rows × {len(data_to_export.columns)} columns")
            
            export_format = st.selectbox(
                "Select Export Format",
                ["CSV", "Excel", "Parquet", "JSON", "Feather"]
            )
            
            filename = st.text_input("Filename (without extension)", value="exported_data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if export_format == "CSV":
                    csv = data_to_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{filename}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        data_to_export.to_excel(writer, index=False, sheet_name='Data')
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"{filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                
                elif export_format == "Parquet":
                    buffer = io.BytesIO()
                    data_to_export.to_parquet(buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Parquet",
                        data=buffer.getvalue(),
                        file_name=f"{filename}.parquet",
                        mime="application/octet-stream",
                        type="primary"
                    )
                
                elif export_format == "JSON":
                    json_str = data_to_export.to_json(orient='records', indent=2)
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"{filename}.json",
                        mime="application/json",
                        type="primary"
                    )
                
                elif export_format == "Feather":
                    buffer = io.BytesIO()
                    data_to_export.to_feather(buffer)
                    
                    st.download_button(
                        label="📥 Download Feather",
                        data=buffer.getvalue(),
                        file_name=f"{filename}.feather",
                        mime="application/octet-stream",
                        type="primary"
                    )
            
            with col2:
                # Save to local directory
                save_path = st.text_input("Or save to local path", value=f"data/processed/{filename}")
                
                if st.button("💾 Save Locally"):
                    try:
                        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                        
                        if export_format == "CSV":
                            data_to_export.to_csv(f"{save_path}.csv", index=False)
                        elif export_format == "Excel":
                            data_to_export.to_excel(f"{save_path}.xlsx", index=False)
                        elif export_format == "Parquet":
                            data_to_export.to_parquet(f"{save_path}.parquet", index=False)
                        elif export_format == "JSON":
                            data_to_export.to_json(f"{save_path}.json", orient='records', indent=2)
                        elif export_format == "Feather":
                            data_to_export.to_feather(f"{save_path}.feather")
                        
                        st.success(f"✅ Saved to {save_path}.{export_format.lower()}")
                    except Exception as e:
                        st.error(f"❌ Save failed: {str(e)}")
    
    # ==================== TAB 5: Data Quality Report ====================
    with tab5:
        data_to_analyze = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.loaded_data
        
        if data_to_analyze is None:
            st.warning("⚠️ Please load data first")
        else:
            st.subheader("Data Quality Report")
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                completeness = (1 - data_to_analyze.isnull().sum().sum() / (len(data_to_analyze) * len(data_to_analyze.columns))) * 100
                st.metric("Completeness", f"{completeness:.1f}%")
            
            with col2:
                uniqueness = (1 - data_to_analyze.duplicated().sum() / len(data_to_analyze)) * 100
                st.metric("Uniqueness", f"{uniqueness:.1f}%")
            
            with col3:
                st.metric("Total Records", f"{len(data_to_analyze):,}")
            
            with col4:
                st.metric("Total Fields", len(data_to_analyze.columns))
            
            # Quality score
            quality_score = (completeness + uniqueness) / 2
            
            st.markdown("### Overall Data Quality Score")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Column-level quality
            st.markdown("### Column-Level Quality Metrics")
            
            quality_df = pd.DataFrame({
                'Column': data_to_analyze.columns,
                'Completeness %': ((1 - data_to_analyze.isnull().sum() / len(data_to_analyze)) * 100).values,
                'Unique Values': data_to_analyze.nunique().values,
                'Uniqueness %': ((data_to_analyze.nunique() / len(data_to_analyze)) * 100).values
            })
            
            quality_df['Quality Score'] = (quality_df['Completeness %'] + quality_df['Uniqueness %']) / 2
            quality_df = quality_df.sort_values('Quality Score', ascending=False)
            
            st.dataframe(quality_df, use_container_width=True)
            
            # Visualize quality by column
            # Convert to native Python types for JSON serialization
            quality_df_plot = quality_df.copy()
            for col in quality_df_plot.columns:
                if pd.api.types.is_numeric_dtype(quality_df_plot[col]):
                    quality_df_plot[col] = quality_df_plot[col].astype(float)
                else:
                    quality_df_plot[col] = quality_df_plot[col].astype(str)
            
            fig = px.bar(quality_df_plot, x='Column', y='Quality Score',
                        title='Data Quality Score by Column',
                        color='Quality Score',
                        color_continuous_scale='RdYlGn')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    show()
