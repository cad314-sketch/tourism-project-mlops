import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prédiction Forfait Bien-être", layout="centered")

st.title("✈️ Prédiction d'Achat du Forfait de Tourisme de Bien-être")
st.write("Cette application prédit si un client est susceptible d'acheter le nouveau forfait de tourisme de bien-être.")

# Path to the saved model
model_path = "tourism_project/deployment/model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error(f"Modèle introuvable à l'emplacement : {model_path}. Veuillez d'abord exécuter l'entraînement du modèle.")
else:
    st.subheader("Informations du client")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        city_tier = st.selectbox("Catégorie de Ville (CityTier)", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        gender = st.selectbox("Genre", ["Male", "Female"])
        marital_status = st.selectbox("État Civil", ["Single", "Married", "Divorced", "Unmarried"])
        designation = st.selectbox("Désignation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Revenu Mensuel", min_value=0, value=20000)

    with col2:
        typeof_contact = st.selectbox("Type de Contact", ["Self Inquiry", "Company Invited"])
        num_person = st.number_input("Nombre de personnes voyageant", min_value=1, max_value=10, value=2)
        pref_stars = st.selectbox("Étoiles d'hôtel préférées", [3, 4, 5])
        num_trips = st.number_input("Nombre de voyages par an", min_value=0, max_value=20, value=2)
        passport = st.selectbox("Détient un Passeport ?", ["Non", "Oui"])
        own_car = st.selectbox("Possède une Voiture ?", ["Non", "Oui"])
        num_children = st.number_input("Nombre d'enfants de moins de 5 ans", min_value=0, max_value=5, value=0)

    st.subheader("Données d'interaction client")
    col3, col4 = st.columns(2)
    with col3:
        pitch_satisfaction = st.slider("Score de satisfaction de la présentation", 1, 5, 3)
        product_pitched = st.selectbox("Produit présenté", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    with col4:
        num_followups = st.number_input("Nombre de suivis", min_value=0, max_value=10, value=3)
        duration_pitch = st.number_input("Durée de la présentation (minutes)", min_value=0, value=15)

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

    if st.button("Prédire l'Achat"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.write("---")
        if prediction == 1:
            st.success(f"🎯 **Le client est susceptible d'acheter le forfait !** (Probabilité : {probability:.2%})")
        else:
            st.warning(f"❌ **Le client est peu susceptible d'acheter le forfait.** (Probabilité d'achat : {probability:.2%})")
