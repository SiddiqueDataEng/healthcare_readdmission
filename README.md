# 🏥 Healthcare Readmission Analytics Platform

**Complete End-to-End ML/DS Solution with Interactive Dashboard**

A comprehensive healthcare analytics platform featuring synthetic EHR data generation, advanced machine learning models, and an interactive Streamlit dashboard for predicting and analyzing 30-day hospital readmissions.

---

## Project Story: From EHR Data to Readmission Decisions

### 1. The problem

Hospital readmissions within 30 days are costly for patients and health systems, and they are also an important quality measure in programs such as the Hospital Readmissions Reduction Program. The practical question for this project is:

> Given information available during an index heart-failure admission, can we estimate whether the patient will be readmitted within 30 days of discharge and present the result in a way that a care team can understand and act on?

The solution must do more than produce a binary prediction. It must assemble a reproducible cohort, avoid using information that would only be known after discharge, handle clinical data quality issues, compare multiple modeling approaches, expose uncertainty and performance, and provide explanations for individual patients.

### 2. Data sources and data itself

The project uses fully synthetic, HIPAA-safe EHR-style data. It is designed to exercise the same integration and modeling workflow as a real clinical data project without exposing real patient information.

The raw source tables are stored in `data/raw/` and are connected through `patient_id` and `encounter_id`:

| Source | Repository snapshot | Important fields | Purpose |
|---|---:|---|---|
| `patients.csv` | 5,000 patients | demographics, insurance, ZIP code, death date | Patient context and competing-risk exclusion |
| `encounters.csv` | 9,970 encounters | admission/discharge dates, diagnosis, disposition, encounter type | Index admissions, readmissions, length of stay, utilization |
| `diagnoses.csv` | 49,894 diagnoses | diagnosis code, type, date | Comorbidity indicators and diagnosis burden |
| `labs.csv` | 99,120 lab results | BNP, sodium, creatinine, potassium, hemoglobin | Clinical severity and physiologic measurements |
| `medications.csv` | 24,968 medication records | medication, class, prescription date, refills | Heart-failure treatment and medication-utilization features |

The generator creates the source data with fixed random seeds for repeatability. The repository also contains processed CSV snapshots in `data/processed/`; the current full-dataset snapshot contains 248 modeling rows after the SQL cohort rules and exclusions are applied. Row counts can change when the synthetic data are regenerated or when the date window is changed.

### 3. Cohort construction and target definition

`sql/sql01_extract_index_admissions.sql` is the central cohort definition. It:

1. Selects inpatient heart-failure admissions using ICD-10 codes `I501`, `I502`, `I503`, and `I509`.
2. Keeps the most recent qualifying admission for each patient in the configured time window.
3. Searches for the next inpatient encounter after discharge.
4. Sets `readmitted_30d = 1` when the next admission occurs within 30 days; otherwise it sets the target to `0`.
5. Excludes patients who died within 30 days because death is a competing outcome for readmission.
6. Joins demographics, diagnoses, labs, medications, and prior utilization into one analytic dataset.

The prediction target is therefore a binary outcome: `readmitted_30d`. Dates, identifiers, the future readmission date, days to readmission, and post-outcome fields are excluded from model features to reduce target leakage.

### 4. Data preparation and feature engineering

The data pipeline in `src/data_pipeline.py` and the model preparation code in `page_modules/ml_models.py` apply these techniques:

- Median imputation for missing laboratory values and consistent missing-value filling during model preparation.
- Categorical conversion and one-hot encoding with train/test column alignment.
- Age bands (`<65`, `65-74`, `75-84`, `85+`) and BNP severity categories.
- A comorbidity score formed from hypertension, diabetes, CAD, COPD, CKD, atrial fibrillation, obesity, and anemia indicators.
- Last available BNP, sodium, creatinine, and hemoglobin values before discharge.
- Heart-failure medication count, total medication count, refill behavior, prior admissions, prior ED visits, length of stay, and weekend discharge.
- Stratified 80/20 train/test splitting with a fixed random state so class proportions remain comparable.
- Feature-name sanitization so generated model columns are accepted consistently by downstream libraries.

The workflow also produces exploratory reports: target balance, numerical distributions, categorical readmission rates, a correlation matrix, and univariate t-tests in `reports/eda/`.

### 5. Techniques applied

The dashboard compares five supervised classification approaches:

| Technique | Role |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Nonlinear ensemble with balanced class weights |
| Gradient Boosting | Sequential tree-based model |
| XGBoost | Regularized gradient-boosted trees |
| LightGBM | Efficient gradient-boosted tree model used as the prediction fallback |

Because readmission is the minority class, the workflow can apply SMOTE to the training data only. Models are evaluated with ROC-AUC, average precision, precision, recall, F1, confusion matrices, Brier score, and calibration curves. This is important because a clinically useful risk score must rank patients well and produce probabilities that are not misleading.

The explainability module uses SHAP for global feature importance, patient-level contribution plots, and feature interactions. The analytics modules add cohort comparisons, statistical tests, risk tiers, and interactive Plotly visualizations.

### 6. The final solution

The final deliverable is an end-to-end Streamlit application in `app.py` with a single workflow:

1. Generate or load the synthetic source data.
2. Extract and label the heart-failure readmission cohort with SQL.
3. Clean, enrich, and split the data into train/test files.
4. Explore outcome patterns and data quality in the dashboard.
5. Train and compare five models, optionally balancing the training class with SMOTE.
6. Select or load a compatible model for real-time scoring.
7. Enter one patient or upload a batch to receive a probability, risk tier, and clinical follow-up guidance.
8. Open SHAP explanations to see why the model assigned that risk.
9. Export scored results for downstream review.

This turns disconnected EHR-style tables into a repeatable clinical analytics workflow: cohort definition, feature engineering, predictive modeling, evaluation, explanation, and operational review in one place.

### 7. Scope and responsible use

The data are synthetic, so this project demonstrates engineering and analytical methods rather than validated clinical performance. It must not be used to make real patient-care decisions without external validation, prospective testing, calibration review, bias and subgroup analysis, privacy controls, clinical governance, and approval for the intended deployment environment. Model output is decision support, not a diagnosis or a replacement for clinician judgment.

---

## 🎯 Quick Start (3 Steps)

### Option 1: One-Click Launch (Recommended)
```cmd
LAUNCH_DASHBOARD.bat
```
This single command will:
1. ✅ Set up environment
2. ✅ Generate data
3. ✅ Launch dashboard

### Option 2: Step-by-Step
```cmd
# Step 1: Setup & Generate Data
START_HERE.bat

# Step 2: Launch Dashboard
run_dashboard.bat
```

**That's it!** The dashboard opens automatically in your browser.

---

## 📊 What You Get

### 🎨 Interactive Dashboard
- **6 Complete Modules** for comprehensive analysis
- **Real-time predictions** with clinical recommendations
- **Advanced ML models** (XGBoost, LightGBM, Random Forest, etc.)
- **SHAP explainability** for model interpretability
- **Custom analytics** and data exploration

### 📁 Complete Dataset
- **5,000 synthetic patients**
- **~10,000 hospital encounters**
- **~50,000 diagnoses**
- **~100,000 lab results**
- **~25,000 medication records**

### 🤖 Machine Learning Pipeline
- **5 trained models** with full evaluation
- **Automated feature engineering**
- **Class imbalance handling** (SMOTE)
- **Model persistence** for deployment
- **Comprehensive metrics** (ROC-AUC, Precision, Recall, F1)

---

## 📋 Dashboard Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| 📊 **Overview & EDA** | Exploratory data analysis | Distributions, correlations, demographics |
| 📈 **Advanced Analytics** | Statistical analysis | T-tests, risk stratification, cohort analysis |
| 🤖 **Machine Learning** | Model training & evaluation | 5 models, ROC curves, feature importance |
| 🔍 **Explainability** | SHAP interpretability | Global/local explanations, interactions |
| 🎯 **Predictions** | Real-time risk assessment | Single patient & batch predictions |
| 📋 **Data Explorer** | Interactive data browser | Filtering, custom charts, quality checks |

---

## 🛠️ System Requirements

- **OS**: Windows (batch scripts) or any OS (Python)
- **Python**: 3.7 or higher
- **RAM**: 2GB+ recommended
- **Disk**: ~100MB for data and dependencies
- **Browser**: Chrome, Firefox, Edge, or Safari

---

## 📦 Installation & Setup

### Automatic Setup (Recommended)
```cmd
LAUNCH_DASHBOARD.bat
```

### Manual Setup
```cmd
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
python src\generate_synthetic_data.py

# 5. Run data pipeline
python src\data_pipeline.py

# 6. Launch dashboard
streamlit run app.py
```

---

## 🎓 Usage Examples

### Example 1: Exploratory Analysis
1. Launch dashboard: `LAUNCH_DASHBOARD.bat`
2. Navigate to "📊 Overview & EDA"
3. Explore patient demographics
4. Analyze readmission patterns
5. Identify key risk factors

### Example 2: Train ML Models
1. Go to "🤖 Machine Learning"
2. Enable SMOTE for class balance
3. Click "Train All Models"
4. Compare model performance
5. Save best model

### Example 3: Predict Patient Risk
1. Go to "🎯 Predictions"
2. Enter patient information
3. Get instant risk assessment
4. Review risk factors
5. Follow clinical recommendations

### Example 4: Understand Predictions
1. Go to "🔍 Model Explainability"
2. Select a patient
3. View SHAP waterfall plot
4. Understand feature contributions
5. Explore feature interactions

---

## 📊 Key Features

### Data Science
- ✅ Comprehensive EDA with 20+ visualizations
- ✅ Statistical significance testing
- ✅ Correlation analysis
- ✅ Risk stratification algorithms
- ✅ Cohort analysis tools

### Machine Learning
- ✅ 5 ML algorithms (Logistic, RF, GB, XGBoost, LightGBM)
- ✅ Automated hyperparameter tuning
- ✅ SMOTE for imbalanced data
- ✅ Cross-validation
- ✅ Model persistence

### Explainability
- ✅ SHAP global feature importance
- ✅ Individual prediction explanations
- ✅ Waterfall plots
- ✅ Force plots
- ✅ Feature interaction analysis

### Predictions
- ✅ Single patient risk assessment
- ✅ Batch predictions (CSV upload)
- ✅ Risk level classification
- ✅ Clinical recommendations
- ✅ Downloadable results

### Visualization
- ✅ Interactive Plotly charts
- ✅ Real-time filtering
- ✅ Custom color schemes
- ✅ Responsive design
- ✅ Export capabilities

---

## 📁 Project Structure

```
healthcare-analytics/
│
├── 🚀 LAUNCH_DASHBOARD.bat        # One-click launcher
├── 🎯 START_HERE.bat              # Setup script
├── 📊 app.py                      # Main dashboard
│
├── pages/                         # Dashboard modules
│   ├── overview.py                # EDA
│   ├── analytics.py               # Advanced analytics
│   ├── ml_models.py               # ML training
│   ├── explainability.py          # SHAP
│   ├── predictions.py             # Predictions
│   └── data_explorer.py           # Data browser
│
├── src/                           # Data pipeline
│   ├── generate_synthetic_data.py # Data generation
│   └── data_pipeline.py           # Data processing
│
├── data/
│   ├── raw/                       # Raw data & database
│   └── processed/                 # Processed datasets
│
├── models/                        # Saved ML models
├── reports/                       # EDA reports
├── sql/                           # SQL queries
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── DASHBOARD_README.md            # Dashboard guide
└── QUICK_START.md                 # Quick start guide
```

---

## 🎨 Dashboard Screenshots

### Overview Module
- Key metrics dashboard
- Distribution analysis
- Correlation heatmaps
- Demographic breakdowns

### ML Module
- Model comparison table
- ROC & PR curves
- Confusion matrices
- Feature importance

### Predictions Module
- Interactive input form
- Risk gauge visualization
- Clinical recommendations
- Batch processing

### Explainability Module
- SHAP summary plots
- Waterfall explanations
- Force plots
- Feature interactions

---

## 🔧 Configuration

### Customize Models
Edit `pages/ml_models.py`:
```python
# Adjust model parameters
models['XGBoost'] = xgb.XGBClassifier(
    n_estimators=200,      # Increase trees
    learning_rate=0.05,    # Lower learning rate
    max_depth=7            # Deeper trees
)
```

### Customize Dashboard
Edit `app.py`:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="🏥",
    layout="wide"
)
```

### Generate More Data
Edit `src/generate_synthetic_data.py`:
```python
# Generate more patients
patients_df = generate_synthetic_patients(n_patients=10000)
```

---

## 📈 Performance Metrics

### Current Results (Synthetic Data)
- **Dataset**: 751 patients
- **Readmission Rate**: 1.33%
- **Best Model**: XGBoost
- **ROC-AUC**: ~0.85-0.90 (typical)
- **Training Time**: <1 minute

### Optimization Tips
1. Use SMOTE for better recall
2. Tune hyperparameters with Optuna
3. Feature engineering for domain knowledge
4. Ensemble methods for robustness

---

## 🚨 Troubleshooting

### Issue: Dashboard won't start
**Solution**: 
```cmd
venv\Scripts\activate.bat
pip install --upgrade streamlit plotly
streamlit run app.py
```

### Issue: Data not found
**Solution**: 
```cmd
START_HERE.bat
```

### Issue: Model training fails
**Solution**: 
- Check available RAM (need 2GB+)
- Reduce dataset size
- Disable SMOTE if memory limited

### Issue: SHAP calculations slow
**Solution**: 
- Reduce sample size in explainability module
- Use faster models (LightGBM)
- Close other applications

### Issue: Port 8501 in use
**Solution**: 
```cmd
streamlit run app.py --server.port 8502
```

---

## 📚 Documentation

- **Quick Start**: `QUICK_START.md`
- **Dashboard Guide**: `DASHBOARD_README.md`
- **Setup Guide**: `SETUP_README.md`
- **API Documentation**: See docstrings in code

---

## 🎯 Use Cases

### Clinical
- ✅ Identify high-risk patients
- ✅ Prioritize follow-up care
- ✅ Optimize discharge planning
- ✅ Reduce readmission rates

### Research
- ✅ Analyze risk factors
- ✅ Test interventions
- ✅ Validate hypotheses
- ✅ Generate insights

### Education
- ✅ Learn ML in healthcare
- ✅ Practice data science
- ✅ Understand model interpretability
- ✅ Explore EHR data

### Quality Improvement
- ✅ Monitor trends
- ✅ Benchmark performance
- ✅ Identify gaps in care
- ✅ Track interventions

---

## 🔐 Data Privacy & Ethics

- ✅ **100% Synthetic Data** - No real patients
- ✅ **Local Processing** - No cloud uploads
- ✅ **Open Source** - Transparent algorithms
- ✅ **Educational Purpose** - Not for clinical use

**Important**: This is a demonstration platform. Do not use for actual clinical decisions without proper validation and regulatory approval.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional ML models
- More visualization types
- Enhanced feature engineering
- Hyperparameter optimization
- Additional data sources
- Mobile responsiveness

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 🆘 Support

### Getting Help
1. Check `TROUBLESHOOTING` section above
2. Review `DASHBOARD_README.md`
3. Check error messages in console
4. Verify system requirements

### Common Questions

**Q: Is this real patient data?**
A: No, all data is synthetic and generated locally.

**Q: Can I use this in production?**
A: This is a demonstration. Clinical use requires validation, testing, and regulatory approval.

**Q: How do I add more features?**
A: Edit `src/generate_synthetic_data.py` and `src/data_pipeline.py`

**Q: Can I deploy this online?**
A: Yes, but ensure HIPAA compliance if using real data.

---

## 🎉 What's Included

### ✅ Complete Data Pipeline
- Synthetic data generation
- SQL-based feature extraction
- Data cleaning & preprocessing
- Train/test splitting

### ✅ Advanced ML Models
- 5 algorithms with full evaluation
- Automated training pipeline
- Model persistence
- Hyperparameter tuning ready

### ✅ Interactive Dashboard
- 6 comprehensive modules
- 50+ visualizations
- Real-time predictions
- Export capabilities

### ✅ Model Explainability
- SHAP integration
- Global & local explanations
- Feature interactions
- Clinical interpretability

### ✅ Production-Ready Scripts
- Automated setup
- Error handling
- Progress tracking
- One-click deployment

---

## 🚀 Next Steps

1. **Run the platform**: `LAUNCH_DASHBOARD.bat`
2. **Explore the data**: Navigate through dashboard modules
3. **Train models**: Use ML module to build predictive models
4. **Make predictions**: Test with sample patients
5. **Understand results**: Use SHAP for interpretability
6. **Customize**: Modify for your specific needs

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| **Complete Setup + Launch** | `LAUNCH_DASHBOARD.bat` |
| **Setup Only** | `START_HERE.bat` |
| **Launch Dashboard** | `run_dashboard.bat` |
| **Generate Data** | `python src\generate_synthetic_data.py` |
| **Run Pipeline** | `python src\data_pipeline.py` |
| **Custom Port** | `streamlit run app.py --server.port 8502` |

---

**Ready to start?** Run `LAUNCH_DASHBOARD.bat` now! 🚀

---

*Built with ❤️ using Python, Streamlit, scikit-learn, XGBoost, SHAP, and Plotly*
