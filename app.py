import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Production App Layout & Page Configurations
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Production-safe Model Loader (cached to prevent reloading on page interactions)
@st.cache_resource
def load_production_model():
    try:
        return joblib.load("model_pipeline.pkl")
    except FileNotFoundError:
        st.error(
            "Model file 'model_pipeline.pkl' not found! Please run your training script (train.py) first to generate it."
        )
        return None

model = load_production_model()

# 3. Define Expected Feature Order
expected_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Zero value cleanup definition array
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# 4. Navigation Control Center Sidebar
st.sidebar.title("🩺 Control Center")
app_mode = st.sidebar.radio(
    "Choose Prediction Mode", 
    ["Single Patient Prediction", "Bulk File Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** Biologically invalid values input as `0` for fields like Glucose, Blood Pressure, "
    "Skin Thickness, Insulin, or BMI are automatically handled by our median imputation pipeline."
)

# ----------------------------------------------------
# NAVIGATION MODE 1: Single Patient Prediction
# ----------------------------------------------------
if app_mode == "Single Patient Prediction":
    st.title("🩺 Diabetes Risk Assessment")
    st.markdown("Enter patient physiological indicators below to calculate individual risk scores.")

    if model is not None:
        with st.form("patient_form"):
            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
                glucose = st.number_input(
                    "Glucose Level (mg/dL)", 
                    min_value=0, max_value=300, value=120, 
                    help="Enter 0 if data is missing"
                )
                blood_pressure = st.number_input(
                    "Diastolic Blood Pressure (mm Hg)", 
                    min_value=0, max_value=150, value=70, 
                    help="Enter 0 if data is missing"
                )
                skin_thickness = st.number_input(
                    "Triceps Skin Fold Thickness (mm)", 
                    min_value=0, max_value=100, value=20, 
                    help="Enter 0 if data is missing"
                )

            with col2:
                insulin = st.number_input(
                    "2-Hour Serum Insulin (mu U/ml)", 
                    min_value=0, max_value=900, value=80, 
                    help="Enter 0 if data is missing"
                )
                bmi = st.number_input(
                    "Body Mass Index (BMI)", 
                    min_value=0.0, max_value=70.0, value=32.0, step=0.1, 
                    help="Enter 0 if data is missing"
                )
                dpf = st.number_input(
                    "Diabetes Pedigree Function Value", 
                    min_value=0.0, max_value=3.0, value=0.5, step=0.01
                )
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=33, step=1)

            submit_button = st.form_submit_button(label="Calculate Risk Score")

        if submit_button:
            # Structuring raw user arrays into production frame structures
            input_data = pd.DataFrame(
                [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                columns=expected_features,
            )

            # Convert 0 to NaN to invoke pipeline's Imputer logic safely
            input_data[zero_cols] = input_data[zero_cols].replace(0, np.nan)

            # Execution logic
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[:, 1][0]

            # Results visualizer dashboard UI components
            st.markdown("---")
            st.subheader("Diagnostic Metrics Output")
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                if prediction[0] == 1:
                    st.error("⚠️ Status: High Risk Predicted")
                else:
                    st.success("✅ Status: Low Risk Predicted")
                    
            with col_metric2:
                st.metric(label="Risk Probability", value=f"{probability:.2%}")
                
            st.progress(float(probability))

# ----------------------------------------------------
# NAVIGATION MODE 2: Bulk File Prediction
# ----------------------------------------------------
elif app_mode == "Bulk File Prediction":
    st.title("📊 Batch Diagnostics & File Upload")
    st.markdown(
        "Upload a `.csv` batch file containing multiple patient profiles for high-throughput evaluation."
    )

    # Built-in Interactive Template Downloader Engine
    st.subheader("📋 Step 1: Download the Template File")
    st.write("Ensure your files match the framework layout matrix before uploading records:")
    
    # Corrected dictionary with explicit arrays
    sample_data = pd.DataFrame({
        "Pregnancies": [6, 1, 1],
        "Glucose": [148, 85, 183],
        "BloodPressure": [72, 66, 64],
        "SkinThickness": [35, 29, 0],       
        "Insulin": [0, 0, 0],               
        "BMI": [33.6, 26.6, 23.3],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.672],
        "Age": [50, 31, 32]
    })
    
    sample_csv = sample_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv,
        file_name="diabetes_bulk_upload_template.csv",
        mime="text/csv",
        help="Download a perfectly configured template matrix populated with sample records."
    )

    st.markdown("---")
    st.subheader("📤 Step 2: Upload Your Production File")

    uploaded_file = st.file_uploader(
        "Upload your patient data file (CSV format only)", type=["csv"]
    )

    if uploaded_file is not None and model is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)

            # Quality Check: verify incoming dataframe strictly maps down onto features
            missing_cols = [col for col in expected_features if col not in bulk_df.columns]

            if missing_cols:
                st.error(
                    f"Schema Mismatch Error! The uploaded file is missing required columns: {missing_cols}"
                )
            else:
                st.success("File framework schema verified. Running classification arrays...")

                # Isolate target evaluation series
                scoring_df = bulk_df[expected_features].copy()

                # Pre-convert raw zeros down to NaNs to prevent mean/median data shifts
                scoring_df[zero_cols] = scoring_df[zero_cols].replace(0, np.nan)

                # Append operational prediction matrix metrics onto the presentation copy
                bulk_df["Predicted_Outcome"] = model.predict(scoring_df)
                bulk_df["Diagnostic_Result"] = bulk_df["Predicted_Outcome"].map({0: "Low Risk", 1: "High Risk"})
                bulk_df["Diabetes_Probability"] = model.predict_proba(scoring_df)[:, 1]

                # Organize structure visualization columns safely
                display_cols = [
                    "Diagnostic_Result",
                    "Diabetes_Probability",
                ] + [c for c in bulk_df.columns if c not in ["Predicted_Outcome", "Diagnostic_Result", "Diabetes_Probability"]]

                # Render analytical records grid layout UI
                st.subheader("Evaluated Patient Output Preview")
                st.dataframe(
                    bulk_df[display_cols].style.background_gradient(
                        subset=["Diabetes_Probability"], cmap="YlOrRd"
                    )
                )

                # Dynamic File Downloader Output Payload
                output_csv = bulk_df.to_csv(index=False).encode("utf-8")
                st.markdown("---")
                st.download_button(
                    label="📥 Download Annotated Predictions File",
                    data=output_csv,
                    file_name="diabetic_batch_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Critical operational error occurred while processing dataframe payload: {e}")
