"""
Data Explorer Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def show(full_data):
    """Display data explorer"""
    
    st.header("📋 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Browse Data", "📊 Custom Analysis", "📈 Data Quality"])
    
    with tab1:
        st.subheader("Dataset Browser")
        
        # Filters
        st.write("**Apply Filters**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age_range = st.slider("Age", 
                                 int(full_data['age'].min()), 
                                 int(full_data['age'].max()),
                                 (int(full_data['age'].min()), int(full_data['age'].max())))
        
        with col2:
            gender_filter = st.multiselect("Gender", 
                                          full_data['gender'].unique().tolist(),
                                          default=full_data['gender'].unique().tolist())
        
        with col3:
            readmit_filter = st.multiselect("Readmission Status",
                                           [0, 1],
                                           default=[0, 1],
                                           format_func=lambda x: "Readmitted" if x == 1 else "Not Readmitted")
        
        with col4:
            discharge_filter = st.multiselect("Discharge Location",
                                             full_data['discharge_location'].unique().tolist(),
                                             default=full_data['discharge_location'].unique().tolist())
        
        # Apply filters
        filtered_data = full_data[
            (full_data['age'] >= age_range[0]) &
            (full_data['age'] <= age_range[1]) &
            (full_data['gender'].isin(gender_filter)) &
            (full_data['readmitted_30d'].isin(readmit_filter)) &
            (full_data['discharge_location'].isin(discharge_filter))
        ]
        
        st.write(f"**Showing {len(filtered_data):,} of {len(full_data):,} patients**")
        
        # Column selector
        all_columns = filtered_data.columns.tolist()
        default_columns = ['age', 'gender', 'readmitted_30d', 'last_bnp', 'length_of_stay', 
                          'comorbidity_score', 'discharge_location']
        
        selected_columns = st.multiselect(
            "Select Columns to Display",
            all_columns,
            default=[col for col in default_columns if col in all_columns]
        )
        
        if selected_columns:
            # Display data
            st.dataframe(
                filtered_data[selected_columns].head(100),
                use_container_width=True
            )
            
            # Download button
            csv = filtered_data[selected_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least one column to display")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        if selected_columns:
            numeric_cols = filtered_data[selected_columns].select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                summary_stats = filtered_data[numeric_cols].describe().T
                summary_stats['missing'] = filtered_data[numeric_cols].isnull().sum()
                summary_stats['missing_pct'] = (summary_stats['missing'] / len(filtered_data) * 100).round(2)
                
                st.dataframe(
                    summary_stats.style.format({
                        'mean': '{:.2f}',
                        'std': '{:.2f}',
                        'min': '{:.2f}',
                        '25%': '{:.2f}',
                        '50%': '{:.2f}',
                        '75%': '{:.2f}',
                        'max': '{:.2f}',
                        'missing_pct': '{:.2f}%'
                    }),
                    use_container_width=True
                )
    
    with tab2:
        st.subheader("Custom Analysis")
        
        st.write("Create custom visualizations and analyses")
        
        # Chart type selector
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Violin Plot"]
        )
        
        col1, col2 = st.columns(2)
        
        # Get numeric and categorical columns
        numeric_cols = full_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = full_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        with col1:
            if chart_type in ["Histogram", "Box Plot", "Violin Plot"]:
                x_col = st.selectbox("Select Variable", numeric_cols)
            elif chart_type == "Scatter Plot":
                x_col = st.selectbox("Select X Variable", numeric_cols)
            else:
                x_col = st.selectbox("Select Category", categorical_cols)
        
        with col2:
            if chart_type == "Scatter Plot":
                y_col = st.selectbox("Select Y Variable", numeric_cols)
            elif chart_type in ["Box Plot", "Violin Plot"]:
                color_col = st.selectbox("Color By", ['readmitted_30d'] + categorical_cols)
            elif chart_type == "Bar Chart":
                y_col = st.selectbox("Select Metric", numeric_cols)
            else:
                color_col = st.selectbox("Color By (Optional)", ['None', 'readmitted_30d'] + categorical_cols)
        
        # Generate chart
        if st.button("Generate Chart"):
            if chart_type == "Histogram":
                color = None if color_col == 'None' else color_col
                fig = px.histogram(
                    full_data,
                    x=x_col,
                    color=color,
                    title=f"Distribution of {x_col}",
                    marginal="box"
                )
            
            elif chart_type == "Box Plot":
                fig = px.box(
                    full_data,
                    x=color_col,
                    y=x_col,
                    color=color_col,
                    title=f"{x_col} by {color_col}"
                )
            
            elif chart_type == "Scatter Plot":
                fig = px.scatter(
                    full_data,
                    x=x_col,
                    y=y_col,
                    color='readmitted_30d',
                    title=f"{y_col} vs {x_col}"
                )
            
            elif chart_type == "Bar Chart":
                agg_data = full_data.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(
                    agg_data,
                    x=x_col,
                    y=y_col,
                    title=f"Average {y_col} by {x_col}"
                )
            
            elif chart_type == "Violin Plot":
                fig = px.violin(
                    full_data,
                    x=color_col,
                    y=x_col,
                    color=color_col,
                    box=True,
                    title=f"{x_col} Distribution by {color_col}"
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr_var1 = st.selectbox("Variable 1", numeric_cols, key='corr1')
        
        with col2:
            corr_var2 = st.selectbox("Variable 2", 
                                    [col for col in numeric_cols if col != corr_var1],
                                    key='corr2')
        
        if st.button("Calculate Correlation"):
            correlation = full_data[corr_var1].corr(full_data[corr_var2])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Pearson Correlation", f"{correlation:.4f}")
                
                if abs(correlation) > 0.7:
                    st.success("Strong correlation")
                elif abs(correlation) > 0.4:
                    st.info("Moderate correlation")
                else:
                    st.warning("Weak correlation")
            
            with col2:
                # Scatter plot
                fig = px.scatter(
                    full_data,
                    x=corr_var1,
                    y=corr_var2,
                    color='readmitted_30d',
                    title=f"{corr_var2} vs {corr_var1}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Data Quality Report")
        
        # Missing values
        st.write("**Missing Values Analysis**")
        
        missing_data = pd.DataFrame({
            'Column': full_data.columns,
            'Missing Count': full_data.isnull().sum(),
            'Missing Percentage': (full_data.isnull().sum() / len(full_data) * 100).round(2)
        }).sort_values('Missing Count', ascending=False)
        
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        
        if len(missing_data) > 0:
            fig = px.bar(
                missing_data,
                x='Column',
                y='Missing Percentage',
                title="Missing Data by Column",
                color='Missing Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_data, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
        
        # Data types
        st.write("**Data Types**")
        
        dtype_df = pd.DataFrame({
            'Column': full_data.columns,
            'Data Type': full_data.dtypes.astype(str),
            'Unique Values': [full_data[col].nunique() for col in full_data.columns],
            'Sample Value': [str(full_data[col].iloc[0]) if len(full_data) > 0 else '' for col in full_data.columns]
        })
        
        st.dataframe(dtype_df, use_container_width=True)
        
        # Outliers detection
        st.write("**Outlier Detection (IQR Method)**")
        
        numeric_cols = full_data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_summary = []
        
        for col in numeric_cols:
            Q1 = full_data[col].quantile(0.25)
            Q3 = full_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = full_data[(full_data[col] < lower_bound) | (full_data[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': len(outliers),
                    'Outlier Percentage': f"{len(outliers)/len(full_data)*100:.2f}%",
                    'Lower Bound': f"{lower_bound:.2f}",
                    'Upper Bound': f"{upper_bound:.2f}"
                })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            st.dataframe(outlier_df, use_container_width=True)
        else:
            st.success("✅ No significant outliers detected!")
        
        # Dataset overview
        st.write("**Dataset Overview**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(full_data):,}")
        with col2:
            st.metric("Total Columns", len(full_data.columns))
        with col3:
            st.metric("Numeric Columns", len(full_data.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Columns", len(full_data.select_dtypes(include=['object', 'category']).columns))
        
        # Memory usage
        memory_usage = full_data.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
