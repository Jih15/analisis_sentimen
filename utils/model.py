import numpy as np
import pandas as pd
import pickle
import io
from datetime import datetime
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
def build_tfidf(X_train: list, max_features: int = 5000) -> tuple:
    """Fit TF-IDF pada data latih dan return vectorizer + matrix."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_vec


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_svm(X_train, y_train):
    model = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_logistic(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        multi_class="auto", random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test, label_names: list) -> dict:
    y_pred = model.predict(X_test)

    labels_present = sorted(set(y_test) | set(y_pred))

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted",
                           labels=labels_present, zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted",
                        labels=labels_present, zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted",
                    labels=labels_present, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=label_names)

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "cm":        cm,
        "y_pred":    y_pred,
        "labels":    label_names,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    X_train: list, y_train: list,
    X_test:  list, y_test:  list,
    label_names: list,
) -> dict:
    """
    Jalankan full pipeline:
    TF-IDF → train SVM & LR → evaluasi pada data uji.
    Return dict berisi hasil kedua model.
    """
    vectorizer, X_train_vec = build_tfidf(X_train)
    X_test_vec = vectorizer.transform(X_test)

    svm_model = train_svm(X_train_vec, y_train)
    lr_model  = train_logistic(X_train_vec, y_train)

    svm_result = evaluate(svm_model, X_test_vec, y_test, label_names)
    lr_result  = evaluate(lr_model,  X_test_vec, y_test, label_names)

    svm_result["nama"] = "Support Vector Machine (SVM)"
    lr_result["nama"]  = "Regresi Logistik"

    return {
        "svm":        svm_result,
        "lr":         lr_result,
        "vectorizer": vectorizer,
        "svm_model":  svm_model,
        "lr_model":   lr_model,
        "n_train":    len(X_train),
        "n_test":     len(X_test),
    }


# ---------------------------------------------------------------------------
# Save model ke bytes (untuk download)
# ---------------------------------------------------------------------------
def save_model(hasil: dict, label_names: list, metadata: dict = None) -> bytes:
    """Simpan model, vectorizer, dan metadata ke bytes (pickle)."""
    bundle = {
        "vectorizer":  hasil["vectorizer"],
        "svm_model":   hasil["svm_model"],
        "lr_model":    hasil["lr_model"],
        "label_names": label_names,
        "metadata": {
            "saved_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_train":     hasil.get("n_train", "?"),
            "svm_accuracy": hasil["svm"]["accuracy"],
            "lr_accuracy":  hasil["lr"]["accuracy"],
            **(metadata or {}),
        },
    }
    buf = io.BytesIO()
    pickle.dump(bundle, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Load model dari bytes (hasil upload)
# ---------------------------------------------------------------------------
def load_model(file_bytes: bytes) -> dict:
    """
    Load bundle dari file .pkl.
    Return dict dengan keys: vectorizer, svm_model, lr_model,
    label_names, metadata.
    """
    buf = io.BytesIO(file_bytes)
    bundle = pickle.load(buf)

    required = {"vectorizer", "svm_model", "lr_model", "label_names"}
    if not required.issubset(bundle.keys()):
        raise ValueError(
            f"File tidak valid. Key yang dibutuhkan: {required}. "
            f"Key yang ada: {set(bundle.keys())}"
        )
    return bundle


# ---------------------------------------------------------------------------
# Evaluasi ulang model yang sudah di-load pada data uji baru
# ---------------------------------------------------------------------------
def evaluate_loaded(bundle: dict, X_test_raw: list, y_test: list,
                    use_stemming: bool = True) -> dict:
    """
    Jalankan preprocessing + evaluasi pada model yang sudah di-load.
    Return dict hasil yang sama formatnya dengan run_pipeline.
    """
    from utils.preprocessing import preprocess_batch

    X_test = preprocess_batch(X_test_raw, use_stemming=use_stemming)
    vectorizer  = bundle["vectorizer"]
    svm_model   = bundle["svm_model"]
    lr_model    = bundle["lr_model"]
    label_names = bundle["label_names"]

    X_test_vec = vectorizer.transform(X_test)

    svm_res = evaluate(svm_model, X_test_vec, y_test, label_names)
    lr_res  = evaluate(lr_model,  X_test_vec, y_test, label_names)

    svm_res["nama"] = "Support Vector Machine (SVM)"
    lr_res["nama"]  = "Regresi Logistik"

    return {
        "svm":        svm_res,
        "lr":         lr_res,
        "vectorizer": vectorizer,
        "svm_model":  svm_model,
        "lr_model":   lr_model,
        "n_train":    bundle["metadata"].get("n_train", "?"),
        "n_test":     len(X_test),
        "label_names": label_names,
        "loaded_from_file": True,
        "metadata":   bundle.get("metadata", {}),
    }
