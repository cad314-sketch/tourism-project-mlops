import os
import pandas as pd

def register_data():
    data_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable à l'emplacement : {data_path}")
    
    df = pd.read_csv(data_path)
    expected_columns = ["ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation"]
    
    print("--- Validation du jeu de données ---")
    print(f"Nombre total de lignes : {len(df)}")
    print(f"Nombre total de colonnes : {len(df.columns)}")
    
    # Vérification des colonnes clés indispensables
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes obligatoires manquantes : {missing_cols}")
    
    print("✓ Validation réussie : Toutes les colonnes indispensables sont présentes.")

if __name__ == "__main__":
    register_data()
