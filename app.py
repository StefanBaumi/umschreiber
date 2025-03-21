# app.py

import streamlit as st
import pickle
import numpy as np
from feature_engineering import extract_features

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def main():
    st.title("KI-Text-Erkenner")
    st.subheader("Erkenne, ob ein Text eher von einer KI oder einem Menschen stammt.")
    st.write("Ein kleines Spaßprojekt – bitte nicht zu ernst nehmen. 😉")

    # Texteingabe
    user_text = st.text_area("Gib deinen Text hier ein:", height=200)

    if st.button("Analysieren"):
        if user_text.strip():
            # Features extrahieren
            feats = extract_features(user_text)
            feats_reshaped = feats.reshape(1, -1)  # Für das Modell als 2D-Array

            # Modell laden und Vorhersage treffen
            model = load_model()
            prediction = model.predict(feats_reshaped)[0]
            pred_prob = model.predict_proba(feats_reshaped)[0]

            # Ergebnis ausgeben
            if prediction == 1:
                st.markdown("### Ergebnis: **Wahrscheinlich KI**")
                st.markdown(
                    f"**Vertrauensscore:** {pred_prob[1]*100:.2f}% für KI vs. {pred_prob[0]*100:.2f}% für Mensch"
                )
            else:
                st.markdown("### Ergebnis: **Wahrscheinlich Mensch**")
                st.markdown(
                    f"**Vertrauensscore:** {pred_prob[0]*100:.2f}% für Mensch vs. {pred_prob[1]*100:.2f}% für KI"
                )
        else:
            st.warning("Bitte gib einen Text ein!")

if __name__ == "__main__":
    main()
