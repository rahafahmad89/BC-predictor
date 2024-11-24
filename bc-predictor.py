import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.markdown(
    """
    <style>
    /* Set the background color */
    .stApp {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("logo.png", width=200)  # Replace "logo.png" with the correct file name

# Title and description
st.title("Breast Cancer Variants Pathogenicity Predictor")
st.markdown("""
This app demonstrates the role of key genomic features in predicting the pathogenicity of breast cancer variants. 
The prediction is powered by a mock RNN model using feature importance derived from analysis of the dataset.
""")

# Input sliders for all features
st.subheader("Input Variant Features")
gerp = st.slider("GERP++ RS (0 to 5)", 0.0, 5.0, 2.5)
phyloP = st.slider("phyloP Mammalian Score (0 to 5)", 0.0, 5.0, 2.5)
cadd = st.slider("CADD PHRED Score (0 to 50)", 0.0, 50.0, 25.0)
fathmm = st.slider("fathmm-MKL Rankscore (0 to 1)", 0.0, 1.0, 0.5)
primate_ai = st.slider("PrimateAI Rankscore (0 to 1)", 0.0, 1.0, 0.5)
deogen2 = st.slider("DEOGEN2 Rankscore (0 to 1)", 0.0, 1.0, 0.5)
ptriplo = st.slider("pTriplo Rankscore (0 to 1)", 0.0, 1.0, 0.5)
gmvp = st.slider("gMVP Rankscore (0 to 1)", 0.0, 1.0, 0.5)
bayesdel = st.slider("BayesDel addAF Rankscore (0 to 1)", 0.0, 1.0, 0.5)

# Weights for features (based on importance analysis)
weights = {
    "GERP++ RS": 0.20,
    "phyloP Mammalian Score": 0.18,
    "CADD PHRED Score": 0.15,
    "fathmm-MKL Rankscore": 0.12,
    "PrimateAI Rankscore": 0.10,
    "DEOGEN2 Rankscore": 0.08,
    "pTriplo Rankscore": 0.07,
    "gMVP Rankscore": 0.06,
    "BayesDel addAF Rankscore": 0.04
}

# Calculate the weighted score
score = (
    gerp * weights["GERP++ RS"] +
    phyloP * weights["phyloP Mammalian Score"] +
    cadd * weights["CADD PHRED Score"] +
    fathmm * weights["fathmm-MKL Rankscore"] +
    primate_ai * weights["PrimateAI Rankscore"] +
    deogen2 * weights["DEOGEN2 Rankscore"] +
    ptriplo * weights["pTriplo Rankscore"] +
    gmvp * weights["gMVP Rankscore"] +
    bayesdel * weights["BayesDel addAF Rankscore"]
)

# Make prediction
threshold = 6  # Mock threshold for pathogenicity
if score > threshold:
    prediction = "Pathogenic"
    bar_color = "red"  # Change bars to red for "Pathogenic"
else:
    prediction = "Benign"
    bar_color = "green"  # Keep bars green for "Benign"

# Display the prediction result
st.subheader("Prediction Result")
st.write(f"**Predicted Pathogenicity:** {prediction}")
st.write(f"**Prediction Score:** {score:.2f}")

# Feature contributions to the prediction
st.subheader("Feature Contributions to the Prediction")
feature_data = pd.DataFrame({
    "Feature": list(weights.keys()),
    "Contribution": [
        gerp * weights["GERP++ RS"],
        phyloP * weights["phyloP Mammalian Score"],
        cadd * weights["CADD PHRED Score"],
        fathmm * weights["fathmm-MKL Rankscore"],
        primate_ai * weights["PrimateAI Rankscore"],
        deogen2 * weights["DEOGEN2 Rankscore"],
        ptriplo * weights["pTriplo Rankscore"],
        gmvp * weights["gMVP Rankscore"],
        bayesdel * weights["BayesDel addAF Rankscore"]
    ]
})

# Plot the feature contributions
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
ax.bar(feature_data["Feature"], feature_data["Contribution"], color=bar_color)
ax.set_ylabel("Contribution to Prediction Score")
ax.set_xlabel("Feature")
ax.set_title("Feature Contributions")
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
st.pyplot(fig)

# Closing note
st.markdown("""
**Note**: This is a mock app designed for demonstration purposes. The weights and thresholds are approximations 
based on feature importance analysis. For real clinical applications, consult validated models and datasets.
""")
