import os
import json
import gzip
import pickle
import pandas as pd
import zipfile
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)

def limpiar_datos(path):
    """Carga y limpia el dataset."""
    with zipfile.ZipFile(path, "r") as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f)
    
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)
    
    return df

# Cargar datos
train_data = limpiar_datos("files/input/train_data.csv.zip")
test_data = limpiar_datos("files/input/test_data.csv.zip")

x_train, y_train = train_data.drop(columns=["default"]), train_data["default"]
x_test, y_test = test_data.drop(columns=["default"]), test_data["default"]

def crear_pipeline():
    """Construye el pipeline de preprocesamiento y modelo."""
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough",
    )
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline

def optimizar_modelo(pipeline, x_train, y_train):
    """Optimiza los hiperparámetros usando validación cruzada."""
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
    }
    scorer = make_scorer(balanced_accuracy_score)
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring=scorer, n_jobs=2, verbose=2)
    grid_search.fit(x_train, y_train)
    return grid_search

def guardar_modelo(model, file_path="files/models/model.pkl.gz"):
    """Guarda el modelo en formato comprimido."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Guardando modelo en: {file_path}")
    with gzip.open(file_path, "wb") as f:
        pickle.dump(model, f)
    if not os.path.exists(file_path):
        print("⚠️ ERROR: El archivo no se guardó correctamente.")
        pickle.dump(model, f)

def evaluar_modelo(model, x_train, y_train, x_test, y_test, file_path="files/output/metrics.json"):
    """Evalúa el modelo y guarda las métricas y matrices de confusión."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    metrics = []
    for dataset, x, y in zip(["train", "test"], [x_train, x_test], [y_train, y_test]):
        y_pred = model.predict(x)
        metrics.append({
            "type": "metrics",
            "dataset": dataset,
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        })
        cm = confusion_matrix(y, y_pred)
        metrics.append({
            "type": "cm_matrix",
            "dataset": dataset,
            "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
        })
    with open(file_path, "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")

# Entrenar y evaluar el modelo
pipeline = build_pipeline()
best_pipeline = optimize_pipeline(pipeline, x_train, y_train)
save_model(best_pipeline)
evaluate_model(best_pipeline, x_train, y_train, x_test, y_test)
