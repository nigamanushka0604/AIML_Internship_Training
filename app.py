import streamlit as st
st.title("House Rent Prediction")
# Input box for Wine Name
wine_name = st.text_input("Enter Wine Name")

# Input box for Wine Price
wine_price = st.number_input("Enter Wine Price", min_value=0.0)

# Display the entered values
st.write("Wine Name:", wine_name)
st.write("Wine Price:", wine_price)