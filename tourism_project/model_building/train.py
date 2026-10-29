import os
import pandas as pd
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    # Chargement des données d'entraînement et de test
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").values.ravel()
    y_test = pd.read_csv("ytest.csv").values.ravel()

    # Identification des colonnes selon leur type
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Prétraitement des données
    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore"), cat_cols)
    )

    # Création du pipeline avec XGBoost
    classifier = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
    pipeline = make_pipeline(preprocessor, classifier)

    # Définition des hyperparamètres pour la recherche par grille
    param_grid = {
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__n_estimators": [50, 100],
        "xgbclassifier__learning_rate": [0.05, 0.1]
    }

    print("Début de la recherche d'hyperparamètres (GridSearchCV)...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)

    # Tracking avec MLflow
    mlflow.set_experiment("Tourism_Prediction")
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Évaluation sur le jeu de test
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Meilleurs paramètres : {best_params}")
        print(f"Précision (Accuracy) : {accuracy:.4f}")
        print("Rapport de classification :\n", report)

        # Enregistrement des paramètres et des métriques dans MLflow
        for param, val in best_params.items():
            mlflow.log_param(param, val)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(best_model, "model")

        # Sauvegarde du modèle final pour le déploiement
        os.makedirs("tourism_project/deployment", exist_ok=True)
        model_path = "tourism_project/deployment/model.joblib"
        joblib.dump(best_model, model_path)
        print(f"✓ Modèle sauvegardé avec succès dans : {model_path}")

if __name__ == "__main__":
    train_model()
