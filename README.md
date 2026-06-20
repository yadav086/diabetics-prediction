# 🩺 Diabetes Risk Prediction App

A production-ready, high-throughput web application built with **Streamlit** and **Scikit-Learn** to predict diabetes risk using patient physiological metrics. The system features an individual clinical assessment calculator alongside an optimized bulk processing pipeline for batch evaluations.

---

## 🚀 Key Architectural Features

*   **Integrated Imputation Pipeline:** Biologically invalid values input as `0` (such as Glucose, Blood Pressure, Skin Thickness, Insulin, or BMI) are dynamically captured and cleaned using a data-safe median strategy.
*   **Mathematical Skew Mitigation:** Applies a specialized `PowerTransformer` (Yeo-Johnson) to stabilize variance and normalize data distributions prior to classification.
*   **Optimized Inference Speed:** Implements explicit model decoupling. The machine learning pipeline is serialized to disk (`model_pipeline.pkl`) and securely cached in memory with `@st.cache_resource` to provide sub-millisecond response times.
*   **Dual Operational Modes:** Easily shift between **Single Patient Diagnostics** and **High-Throughput Batch Processing** with automated `.csv` layout validation.

---

## 🛠️ Installation & Setup

Follow these quick commands to spin up the application on your local machine:

### 1. Environment Configuration
Clone your project repository, open your terminal, and install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 2. File Architecture
Ensure your local project directory structure is organized as follows:
```text
├── diabetes.csv            # Original training dataset 
├── train.py                # Model training and export architecture
├── app.py                  # Streamlit production front-end code
└── README.md               # Documentation guide
```

### 3. Step-by-Step Execution Guide

**Step A: Train the Production Pipeline**  
Run the training script to clean the training data, fit the preprocessing transformers, build the balanced classification model, and export the binary file:
```bash
python train.py
```
*Expected Output:* `Production pipeline saved successfully as 'model_pipeline.pkl'!`

**Step B: Launch the Web Interface**  
Boot up the interactive front-end dashboard on your local hosting server:
```bash
streamlit run app.py
```

---

## 📊 Application Dashboard Manual

### 🔹 Mode 1: Single Patient Prediction
*   **Action:** Input individual clinical measurements (Age, BMI, Glucose, etc.) using interactive sliders and numeric submission fields.
*   **Pipeline Logic:** Missing values can be safely entered as `0`. The framework converts them into `NaN` and applies the median value calculated during model training.
*   **Result:** Displays an instant categorical diagnostic response status accompanied by a calculated risk percentage tracker.

### 🔹 Mode 2: Bulk File Prediction
*   **Action:** Download the built-in, perfectly structured sample data template (`diabetes_bulk_upload_template.csv`) with a single click.
*   **Processing Engine:** Add your custom database profiles into the template framework and re-upload the spreadsheet.
*   **Output Matrix:** The system cross-references database schemas, runs the entire matrix through the prediction pipeline, color-codes risk concentrations, and generates a downloadable annotated predictions spreadsheet file.

---

## 🧬 Machine Learning Pipeline Blueprint

The underlying architecture leverages a robust **Logistic Regression** classifier specifically tuned for health-data imbalances:
1.  **Imputation Layer:** `SimpleImputer(strategy='median', missing_values=np.nan)`
2.  **Normalization Layer:** `PowerTransformer(method='yeo-johnson')`
3.  **Classification Layer:** `LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', random_state=43)`
