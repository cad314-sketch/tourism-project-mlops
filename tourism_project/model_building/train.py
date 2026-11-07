import os
import pandas as pd
import joblib
import mlflow
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    # MLflow configuration with robust local fallback
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Tourism_Prediction")
    except Exception as e:
        print("Note: MLflow server unreachable, local recording in 'mlruns'.")
        mlflow.set_tracking_uri("")
        mlflow.set_experiment("Tourism_Prediction")

    # Loading training and test data
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").values.ravel()
    y_test = pd.read_csv("ytest.csv").values.ravel()

    # Column identification according to their type
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Data preprocessing
    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore"), cat_cols)
    )

    # Creating the pipeline with XGBoost
    classifier = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
    pipeline = make_pipeline(preprocessor, classifier)

    # Definition of hyperparameters for grid search
    param_grid = {
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__n_estimators": [50, 100],
        "xgbclassifier__learning_rate": [0.05, 0.1]
    }

    print("Beginning of the hyperparameter search (GridSearchCV)...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)

    try:
        with mlflow.start_run():
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Evaluation of the test game
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print(f"Meilleurs paramètres : {best_params}")
            print(f"Précision (Accuracy) : {accuracy:.4f}")
            print("Rapport de classification :\n", report)

            # Recording in MLflow
            for param, val in best_params.items():
                mlflow.log_param(param, val)
            mlflow.log_metric("accuracy", accuracy)

            # Signature generation and example entry
            input_example = X_train.head(5)
            signature = infer_signature(X_train, best_model.predict(X_train))

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Saving the final model for deployment
            os.makedirs("tourism_project/deployment", exist_ok=True)
            model_path = "tourism_project/deployment/model.joblib"
            joblib.dump(best_model, model_path)
            print(f"✓ Model successfully saved locally in : {model_path}")
    except Exception as run_error:
        print(f"Note: MLflow experiment failed ({run_error}). Local execution and backup only.")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        os.makedirs("tourism_project/deployment", exist_ok=True)
        model_path = "tourism_project/deployment/model.joblib"
        joblib.dump(best_model, model_path)
        print(f"✓ Model successfully saved locally : {model_path}")

if __name__ == "__main__":
    train_model()
