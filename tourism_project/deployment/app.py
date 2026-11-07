import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Wellness Package Prediction, layout="centered")

st.title("✈️ Wellness Tourism Package Purchase Prediction")
st.write("This application predicts whether a customer is likely to purchase the new wellness tourism package.")

# Path to the saved model
model_path = "tourism_project/deployment/model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error(f"Model not found at location : {model_path}. Please run the model training first.")
else:
    st.subheader("Customer information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        city_tier = st.selectbox("City Category (CityTier)", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)

    with col2:
        typeof_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
        num_person = st.number_input("Number of people travelling", min_value=1, max_value=10, value=2)
        pref_stars = st.selectbox("Favorite hotel stars", [3, 4, 5])
        num_trips = st.number_input("Number of trips per year", min_value=0, max_value=20, value=2)
        passport = st.selectbox("Does he have a passport?", ["No", "Yes"])
        own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
        num_children = st.number_input("Number of children under 5 years old", min_value=0, max_value=5, value=0)

    st.subheader("Customer interaction data")
    col3, col4 = st.columns(2)
    with col3:
        pitch_satisfaction = st.slider("Presentation satisfaction score", 1, 5, 3)
        product_pitched = st.selectbox("Product presented", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    with col4:
        num_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=3)
        duration_pitch = st.number_input("Presentation duration (minutes)", min_value=0, value=15)

    # Prepare the input dictionary mapping exactly to expected dataset features
    input_data = {
        "Age": [age],
        "TypeofContact": [typeof_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [num_person],
        "NumberOfFollowups": [num_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [pref_stars],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [num_trips],
        "Passport": [1 if passport == "Oui" else 0],
        "PitchSatisfactionScore": [pitch_satisfaction],
        "OwnCar": [1 if own_car == "Oui" else 0],
        "NumberOfChildrenVisiting": [num_children],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    }

    input_df = pd.DataFrame(input_data)

    if st.button("Predicting the Purchase"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.write("---")
        if prediction == 1:
            st.success(f"🎯 **The customer is likely to purchase the package!** (Probability : {probability:.2%})")
        else:
            st.warning(f"❌ **
The customer is unlikely to purchase the package.** (Probability of purchase) : {probability:.2%})")
