import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

st.title("🍷 Wine Quality Predictor")
st.write("Enter the chemical properties of the wine below to get a predicted quality score.")

# Load the trained pipeline (scaler + model bundled together)
@st.cache_resource
def load_pipeline():
    return joblib.load("wine_quality_pipeline.pkl")

pipeline = load_pipeline()

st.subheader("Wine Properties")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.70, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.5, value=0.00, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076, step=0.001, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0, step=1.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=34.0, step=1.0)
    density = st.number_input("Density", min_value=0.9000, max_value=1.1000, value=0.9978, step=0.0001, format="%.4f")
    pH = st.number_input("pH", min_value=2.5, max_value=4.5, value=3.51, step=0.01)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.5, value=0.56, step=0.01)
    alcohol = st.number_input("Alcohol (%)", min_value=5.0, max_value=20.0, value=9.4, step=0.1)
    #wine_type = st.selectbox("Wine Type", ["red", "white"])

st.divider()

if st.button("Predict Quality", type="primary", use_container_width=True):
    new_wine = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol],
    })

    prediction = pipeline.predict(new_wine)

    st.success(f"### Predicted Quality Score: {prediction[0]:.2f} / 10")

    with st.expander("See input details"):
        st.dataframe(new_wine, use_container_width=True)