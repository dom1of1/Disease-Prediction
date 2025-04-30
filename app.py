import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and symptom list
with open('models/trained_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('models/symptoms.pkl', 'rb') as file:
    symptoms = pickle.load(file)

# User input
selected_symptoms = st.multiselect("Select your symptoms:", symptoms)

if st.button("Predict Disease"):
    if selected_symptoms:
        # Create input vector
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
        input_df = pd.DataFrame([input_vector], columns=symptoms)

        # Predict probabilities
        probabilities = model.predict_proba(input_df)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = model.classes_[top_indices]
        top_probs = probabilities[top_indices]

        st.subheader("Top 3 Predicted Diseases:")
        for disease, prob in zip(top_diseases, top_probs):
            st.write(f"{disease}: {prob*100:.2f}%")
    else:
        st.warning("Please select at least one symptom.")