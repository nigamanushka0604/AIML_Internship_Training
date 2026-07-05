import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Rent Predictor", layout="centered")

st.title("House Rent Predictor")
st.write("Enter the details of the house below to get a predicted monthly rent.")

@st.cache_resource
def load_pipeline():
    return joblib.load("house_rent_pipeline.pkl")

pipeline = load_pipeline()

st.subheader("House Details")

col1, col2 = st.columns(2)

with col1:
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
    size = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1200, step=50)
    bathroom = st.number_input("Bathroom", min_value=1, max_value=10, value=2, step=1)
    area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])

with col2:
    city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"])
    furnishing_status = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
    tenant_preferred = st.selectbox("Tenant Preferred", ["Family", "Bachelors", "Bachelors/Family"])
    point_of_contact = st.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])

st.divider()

if st.button("Predict Rent", type="primary", use_container_width=True):
    new_house = pd.DataFrame({
        'BHK': [bhk],
        'Size': [size],
        'Area Type': [area_type],
        'City': [city],
        'Furnishing Status': [furnishing_status],
        'Tenant Preferred': [tenant_preferred],
        'Bathroom': [bathroom],
        'Point of Contact': [point_of_contact]
    })

    prediction = pipeline.predict(new_house)

    st.success(f"### Predicted Rent: ₹{prediction[0]:,.2f} / month")

    with st.expander("See input details"):
        st.dataframe(new_house, use_container_width=True)
        
        