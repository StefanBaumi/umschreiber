# train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from feature_engineering import extract_features

def main():
    # 1. Daten laden (CSV mit Spalten: text, label)
    df = pd.read_csv("data.csv")  # Label: "KI" oder "Mensch"
    
    # 2. Features extrahieren und Labels kodieren (KI=1, Mensch=0)
    X = []
    y = []
    for _, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        X.append(extract_features(text))
        y.append(1 if label == "KI" else 0)
    
    # Feature-Namen entsprechend der Reihenfolge in extract_features
    feature_names = [
        "avg_sent_len",
        "var_sent_len",
        "repeated_ratio",
        "flesch_reading_ease",
        "type_token_ratio",
        "filler_word_ratio",
        "personal_pronoun_ratio",
        "passive_indicator_ratio",
        "emoji_ratio"
    ]
    
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y)

    # 3. Trainings-/Testsplit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Modell erstellen & trainieren
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 5. Auswertung
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test-Genauigkeit: {acc:.2f}")

    # 6. Modell speichern
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Modell wurde in model.pkl gespeichert.")

if __name__ == "__main__":
    main()
