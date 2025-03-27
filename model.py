import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, fbeta_score
from scipy.special import expit

pd.options.mode.chained_assignment = None


def open_data(path="data/clients.csv"):
    """Reads df from given path"""
    return pd.read_csv(path, index_col=0)


def preprocess_data(df: pd.DataFrame):
    categorical = [
        "GENDER",
        "MARITAL_STATUS",
        "SOCSTATUS_WORK_FL",
        "SOCSTATUS_PENS_FL",
        "FL_PRESENCE_FL",
    ]
    numerical = [
        "AGE",
        "CHILD_TOTAL",
        "DEPENDANTS",
        "OWN_AUTO",
        "PERSONAL_INCOME",
        "TOTAL_LOANS",
        "CLOSED_LOANS",
        "WORK_YEARS",
    ]

    X = df[categorical + numerical]
    y = df["TARGET"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return X, y, numerical, categorical, preprocessor


def fit_and_save_model(X_train, X_val, y_train, y_val, preprocessor):
    """Fits RidgeClassifier model and returns predictions"""
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    y_pred_proba = expit(model.decision_function(X_val))
    return model, y_pred_proba


def dump_model(model, path="data/model.pkl"):
    """Saves the entire model"""
    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Полная модель сохранена в {path}")


def search_threshold(y_true, y_pred_proba):
    """Finds optimal threshold using F2-score"""
    thresholds = np.arange(0.1, 1.0, 0.02)
    return max(thresholds, key=lambda t: fbeta_score(y_true, y_pred_proba >= t, beta=2))


def save_threshold(threshold, path="data/best_threshold.txt"):
    """Saves the optimal threshold"""
    with open(path, "w") as f:
        f.write(str(threshold))
    print(f"Оптимальный порог сохранён в {path}")


def save_importances(
    model, preprocessor, numerical, categorical, path="data/importances.csv"
):
    """Saves sorted feature weights"""
    feature_names = list(
        preprocessor.named_transformers_["num"].get_feature_names_out(numerical)
    ) + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical))
    coefs = model.named_steps["classifier"].coef_.flatten()

    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df.sort_values("Abs_Coefficient", ascending=False, inplace=True)
    coef_df.drop(columns=["Abs_Coefficient"], inplace=True)
    coef_df.to_csv(path, index=False)
    print(f"Importances saved to {path}")


def load_model(file_path="data/model.pkl"):
    """Loads the full model"""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_threshold(file_path="data/best_threshold.txt"):
    """Loads the saved threshold"""
    with open(file_path, "r") as f:
        return float(f.read())


def predict_on_input(df, model, threshold):
    """Returns prediction and probability using the optimal threshold"""
    proba = expit(model.decision_function(df))[0]
    pred = int(proba >= threshold)
    return pred, proba


def evaluate_model(y_true, y_pred_proba, threshold):
    """Evaluates the model using various metrics"""
    y_pred_thresholded = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_thresholded)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    recall = recall_score(y_true, y_pred_thresholded)
    f2 = fbeta_score(y_true, y_pred_thresholded, beta=2)

    return accuracy, roc_auc, recall, f2


def load_threshold(file_path="data/best_threshold.txt"):
    """Загружает сохранённый порог"""
    with open(file_path, "r") as f:
        return float(f.read())


if __name__ == "__main__":
    # Загрузка данных и подготовка
    df = open_data()
    X, y, numerical, categorical, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучение модели
    model, y_pred_proba = fit_and_save_model(
        X_train, X_test, y_train, y_test, preprocessor
    )
    dump_model(model)

    # Поиск оптимального порога
    best_threshold = search_threshold(y_test, y_pred_proba)
    print(f"Оптимальный порог: {best_threshold}")
    save_threshold(best_threshold)

    # Оценка модели с использованием лучшего порога
    accuracy, roc_auc, recall, f2 = evaluate_model(y_test, y_pred_proba, best_threshold)

    # Вывод финальных метрик
    print("Финальные метрики модели:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F2-Score: {f2:.4f}")

    save_importances(model, preprocessor, numerical, categorical)
