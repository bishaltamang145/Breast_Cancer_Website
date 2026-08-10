import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------------------------------------------
# Load model, scaler, and feature stats
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    stats = pd.read_csv("feature_stats.csv", index_col=0)
    return model, scaler, stats

model, scaler, stats = load_artifacts()
feature_names = stats.index.tolist()

# Group features by mean / se / worst for a cleaner sidebar
groups = {
    "Mean": [f for f in feature_names if f.endswith("_mean")],
    "Standard Error": [f for f in feature_names if f.endswith("_se")],
    "Worst": [f for f in feature_names if f.endswith("_worst")],
}

# ------------------------------------------------------------------
# Sidebar: input sliders
# ------------------------------------------------------------------
st.sidebar.header("Cell Nuclei Measurements")
input_mode = st.sidebar.radio(
    "Input method",
    options=["Upload file", "Sliders", "Number boxes"],
    horizontal=False
)

input_data = {}
uploaded_df = None

if input_mode == "Upload file":
    st.sidebar.write("Upload a CSV or Excel file with the measurement columns.")

    with open("sample_report_template.csv", "rb") as f:
        st.sidebar.download_button(
            "📄 Download sample template",
            data=f,
            file_name="sample_report_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.sidebar.file_uploader(
        "Upload report (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)

            # Drop columns the model doesn't use, if present
            uploaded_df = uploaded_df.drop(
                columns=[c for c in ["id", "Unnamed: 32", "diagnosis"] if c in uploaded_df.columns],
                errors="ignore"
            )

            missing = [c for c in feature_names if c not in uploaded_df.columns]
            if missing:
                st.sidebar.error(f"Missing {len(missing)} required column(s): {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
                uploaded_df = None
            else:
                uploaded_df = uploaded_df[feature_names]
                st.sidebar.success(f"Loaded {len(uploaded_df)} row(s).")
        except Exception as e:
            st.sidebar.error(f"Couldn't read file: {e}")
            uploaded_df = None
    else:
        st.sidebar.info("No file uploaded yet — showing dataset averages below.")
        input_data = {col: float(stats.loc[col, "mean"]) for col in feature_names}

else:
    st.sidebar.write("Adjust the values below (defaults set to dataset averages).")
    for group_name, cols in groups.items():
        with st.sidebar.expander(group_name, expanded=(group_name == "Mean")):
            for col in cols:
                row = stats.loc[col]
                label = col.replace("_mean", "").replace("_se", "").replace("_worst", "").replace("_", " ").title()
                step = (float(row["max"]) - float(row["min"])) / 200
                if input_mode == "Sliders":
                    input_data[col] = st.slider(
                        label,
                        min_value=float(row["min"]),
                        max_value=float(row["max"]),
                        value=float(row["mean"]),
                        step=step,
                        key=f"{col}_slider"
                    )
                else:
                    input_data[col] = st.number_input(
                        label,
                        min_value=float(row["min"]),
                        max_value=float(row["max"]),
                        value=float(row["mean"]),
                        step=step,
                        format="%.5f",
                        key=f"{col}_number"
                    )

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
st.title("🩺 Breast Cancer Diagnosis Predictor")
st.write(
    "This app uses a **Support Vector Machine (SVM)** classifier trained on the "
    "Wisconsin Diagnostic Breast Cancer dataset to predict whether a tumor is "
    "**benign** or **malignant** based on cell nuclei measurements from a biopsy."
)
st.info(
    "⚠️ This tool is for educational/demo purposes only and is **not** a substitute "
    "for professional medical diagnosis."
)

col1, col2 = st.columns([2, 1])

if input_mode == "Upload file" and uploaded_df is not None:
    # ---------------- Batch mode ----------------
    with col1:
        st.subheader(f"Uploaded Data ({len(uploaded_df)} row(s))")
        st.dataframe(uploaded_df, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        if st.button("🔍 Analyze Report", use_container_width=True, type="primary"):
            X_scaled = scaler.transform(uploaded_df)
            predictions = model.predict(X_scaled)
            scores = model.decision_function(X_scaled)

            results = uploaded_df.copy()
            results.insert(0, "Diagnosis", np.where(predictions == 1, "Malignant", "Benign"))
            results.insert(1, "Decision Score", scores.round(3))

            st.session_state["batch_results"] = results

        if "batch_results" in st.session_state:
            results = st.session_state["batch_results"]
            n_malignant = (results["Diagnosis"] == "Malignant").sum()
            n_benign = (results["Diagnosis"] == "Benign").sum()
            st.metric("Malignant", n_malignant)
            st.metric("Benign", n_benign)

    if "batch_results" in st.session_state:
        st.divider()
        st.subheader("Results")
        st.dataframe(
            st.session_state["batch_results"][["Diagnosis", "Decision Score"]],
            use_container_width=True
        )
        csv_out = st.session_state["batch_results"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv_out,
            file_name="diagnosis_results.csv",
            mime="text/csv"
        )

else:
    # ---------------- Single-patient mode (sliders / number boxes / default) ----------------
    with col1:
        st.subheader("Input Summary")
        input_df = pd.DataFrame([input_data])[feature_names]
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    with col2:
        st.subheader("Prediction")

        if st.button("🔍 Predict Diagnosis", use_container_width=True, type="primary"):
            X_scaled = scaler.transform(input_df)
            prediction = model.predict(X_scaled)[0]
            score = model.decision_function(X_scaled)[0]  # signed distance to hyperplane

            if prediction == 1:
                st.error("### Result: Malignant")
            else:
                st.success("### Result: Benign")

            st.metric("Decision Score", f"{score:.3f}")
            st.caption(
                "Signed distance from the SVM decision boundary. "
                "Positive → Malignant side, Negative → Benign side. "
                "Larger magnitude means the point sits farther from the boundary "
                "(i.e. a more confident separation)."
            )

            max_range = 6.0
            clipped = max(min(score, max_range), -max_range)
            gauge_df = pd.DataFrame({"Decision Score": [clipped]}, index=["Benign ← 0 → Malignant"])
            st.bar_chart(gauge_df)
        else:
            st.write("Provide input and click **Predict Diagnosis**.")

st.divider()
st.caption(
    "Model: Linear-kernel SVM · Trained on 569 samples from the Wisconsin "
    "Diagnostic Breast Cancer dataset · Test accuracy ≈ 96.5%"
)
