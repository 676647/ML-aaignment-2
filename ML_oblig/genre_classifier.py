#!/usr/bin/env python3
"""
MUSIC GENRE CLASSIFICATION — pure Python script

Gjør følgende:
1) Leser Spotify-genre-datasett (train.csv fra Kaggle-lenken i beskrivelsen)
2) Renser/forenkler features, slår sammen subgenrer -> 10 hovedgenrer
3) Nedprøver store klasser (maks 5x minste) for å redusere skjevhet
4) Splitter i train/val/test
5) Trener LogisticRegression baseline (class_weight='balanced')
6) Evaluerer og lagrer:
   - klassifikasjonsrapporter (val & test)
   - forvirringsmatriser (som PNG)
   - korrelasjonsmatrise (PNG)
   - trenet modell + scaler + labelencoder (joblib)

Kjøring:
    python genre_classifier.py --data_path /path/to/train.csv --out_dir ./GenreDetectorV2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42


# -----------------------------
# Konfig / mapping av sjangre
# -----------------------------
MAIN_GENRES = ["pop", "rock", "hip-hop", "electronic", "r-n-b",
               "jazz", "classical", "country", "folk", "reggae"]

GENRE_GROUPING: Dict[str, str] = {
    # Pop
    "cantopop": "pop", "j-pop": "pop", "k-pop": "pop",
    "pop-film": "pop", "pop": "pop", "mandopop": "pop",

    # Rock
    "alt-rock": "rock", "alternative": "rock", "hard-rock": "rock",
    "grunge": "rock", "metal": "rock", "metalcore": "rock",
    "emo": "rock", "punk": "rock", "j-rock": "rock", "heavy-metal": "rock",

    # Hip-Hop
    "hip-hop": "hip-hop",

    # Electronic
    "ambient": "electronic", "breakbeat": "electronic", "chill": "electronic",
    "club": "electronic", "house": "electronic", "techno": "electronic",
    "edm": "electronic", "dubstep": "electronic", "trance": "electronic",

    # RnB
    "r-n-b": "r-n-b", "soul": "r-n-b",

    # Jazz / Blues
    "jazz": "jazz", "blues": "jazz",

    # Classical (inkl noen som ofte klassifiseres nært)
    "classical": "classical", "opera": "classical", "tango": "classical",

    # Country
    "country": "country", "honky-tonk": "country", "sertanejo": "country",

    # Folk
    "folk": "folk", "acoustic": "folk", "singer-songwriter": "folk",
    "songwriter": "folk", "bluegrass": "folk",

    # Reggae / latin-derivater som ofte rangeres nært reggae i settet
    "reggae": "reggae", "reggaeton": "reggae", "dancehall": "reggae",
    "ska": "reggae",
}


FEATURE_COLUMNS = [
    "duration_ms", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature"
]

DROP_COLUMNS = [
    "Unnamed: 0", "track_id", "artists", "album_name", "track_name",
    "popularity", "explicit"
]


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


# -----------------------------
# I/O & Utils
# -----------------------------
def ensure_dirs(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for sub in ["models", "plots", "reports"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# -----------------------------
# Dataforberedelse
# -----------------------------
def load_and_prepare(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    # dropp kolonner som ikke skal brukes til modell
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")

    # sikrer at target-kolonne heter "genre" (i rådata heter den ofte "track_genre")
    if "track_genre" in df.columns and "genre" not in df.columns:
        df = df.rename(columns={"track_genre": "genre"})

    # bare bevar relevante features + genre
    keep = [c for c in FEATURE_COLUMNS if c in df.columns] + ["genre"]
    df = df[keep].copy()

    # type-sikkerhet
    numeric_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # dropp rader med NaN i features eller manglende genre
    df = df.dropna(subset=numeric_cols + ["genre"]).reset_index(drop=True)
    return df


def map_and_filter_genres(df: pd.DataFrame) -> pd.DataFrame:
    # Lag en mapping som sender alt ukjent til "DELETE"
    final_mapping = GENRE_GROUPING.copy()
    for g in df["genre"].unique():
        if g not in final_mapping:
            final_mapping[g] = "DELETE"

    df = df.copy()
    df["target"] = df["genre"].map(final_mapping)
    df = df[df["target"] != "DELETE"].drop(columns=["genre"]).reset_index(drop=True)
    return df


def downsample_by_ratio(df: pd.DataFrame, max_ratio: int = 5) -> pd.DataFrame:
    """Nedprøv store klasser slik at største klasse <= max_ratio * minste klasse."""
    counts = df["target"].value_counts()
    size_min = counts.min()
    size_max = size_min * max_ratio

    parts = []
    for genre, count in counts.items():
        part = df[df["target"] == genre]
        if count > size_max:
            part = part.sample(n=size_max, random_state=RANDOM_STATE)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle
    return out


def split_data(df: pd.DataFrame) -> DatasetSplits:
    X = df.drop(columns=["target"])
    y = df["target"]

    # 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # 25% av resten til val => 0.25 * 0.80 = 0.20 totalt => 60/20/20
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


# -----------------------------
# Visualisering (valgfritt)
# -----------------------------
def plot_corr(df_features: pd.DataFrame, out_path: str) -> None:
    corr = df_features.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, classes: List[str], title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Trening & evaluering
# -----------------------------
def train_baseline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[LogisticRegression, StandardScaler, LabelEncoder]:
    # Label-encoder for klassenavn
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Skaler numeriske features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # Logistic Regression baseline
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=None,
        random_state=RANDOM_STATE,
        multi_class="auto",
    )
    clf.fit(X_train_sc, y_train_enc)
    return clf, scaler, le


def evaluate_split(
    clf: LogisticRegression,
    scaler: StandardScaler,
    le: LabelEncoder,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float | str]:
    X_sc = scaler.transform(X)
    y_true = y.values
    y_pred_enc = clf.predict(X_sc)
    y_pred = le.inverse_transform(y_pred_enc)

    report = classification_report(y_true, y_pred, digits=4)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    return {"report": report, "accuracy": acc, "f1_macro": f1_macro, "y_pred": y_pred}


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Music Genre Classification (pure Python)")
    ap.add_argument("--data_path", required=True, help="Path to train.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--no-plots", action="store_true", help="Disable saving plots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)

    print("1) Leser og forbereder data ...")
    df_raw = load_and_prepare(args.data_path)

    print("   Rader før genre-mapping:", len(df_raw))
    df = map_and_filter_genres(df_raw)
    print("   Rader etter genre-mapping:", len(df))

    # nedprøving
    print("2) Nedprøver store klasser (maks 5x minste) ...")
    df_final = downsample_by_ratio(df, max_ratio=5)
    print("   Rader etter nedprøving:", len(df_final))

    # valgfri korrelasjonsmatrise
    if not args.no_plots:
        plot_corr(
            df_final.drop(columns=["target"]),
            os.path.join(args.out_dir, "plots", "correlation_matrix.png"),
        )

    # Splitting
    print("3) Splitter i train/val/test ...")
    splits = split_data(df_final)
    print(f"   Train: {len(splits.X_train)} | Val: {len(splits.X_val)} | Test: {len(splits.X_test)}")

    # Tren baseline
    print("4) Trener Logistic Regression baseline ...")
    clf, scaler, le = train_baseline(splits.X_train, splits.y_train)

    # Eval val
    print("5) Evaluerer på val ...")
    val_metrics = evaluate_split(clf, scaler, le, splits.X_val, splits.y_val)
    save_text(
        os.path.join(args.out_dir, "reports", "val_report.txt"),
        val_metrics["report"]
        + f"\n\nAccuracy: {val_metrics['accuracy']:.4f}"
        + f"\nF1-macro: {val_metrics['f1_macro']:.4f}\n",
    )
    print("   Val Accuracy:", f"{val_metrics['accuracy']:.4f}")
    print("   Val F1-macro:", f"{val_metrics['f1_macro']:.4f}")

    # Eval test
    print("6) Evaluerer på test ...")
    test_metrics = evaluate_split(clf, scaler, le, splits.X_test, splits.y_test)
    save_text(
        os.path.join(args.out_dir, "reports", "test_report.txt"),
        test_metrics["report"]
        + f"\n\nAccuracy: {test_metrics['accuracy']:.4f}"
        + f"\nF1-macro: {test_metrics['f1_macro']:.4f}\n",
    )
    print("   Test Accuracy:", f"{test_metrics['accuracy']:.4f}")
    print("   Test F1-macro:", f"{test_metrics['f1_macro']:.4f}")

    # Forvirringsmatriser
    if not args.no_plots:
        for split_name, (X, y_true, y_pred) in {
            "val": (splits.X_val, splits.y_val, val_metrics["y_pred"]),
            "test": (splits.X_test, splits.y_test, test_metrics["y_pred"]),
        }.items():
            labels_sorted = sorted(list(set(y_true)))
            cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
            plot_confusion(
                cm,
                labels_sorted,
                title=f"Confusion Matrix ({split_name})",
                out_path=os.path.join(args.out_dir, "plots", f"cm_{split_name}.png"),
            )

    # Lagre artefakter
    print("7) Lagrer modell og preprocessors ...")
    joblib.dump(clf, os.path.join(args.out_dir, "models", "logreg_model.joblib"))
    joblib.dump(scaler, os.path.join(args.out_dir, "models", "scaler.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "models", "labelencoder.joblib"))

    # Lagre metadata (features, mapping, etc.)
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "main_genres": MAIN_GENRES,
        "genre_grouping_size": len(GENRE_GROUPING),
        "random_state": RANDOM_STATE,
        "samples_final": int(len(df_final)),
        "class_distribution": df_final["target"].value_counts().to_dict(),
    }
    save_text(os.path.join(args.out_dir, "reports", "metadata.json"), json.dumps(meta, indent=2))
    print("Ferdig!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAvbrutt av bruker.", file=sys.stderr)
        sys.exit(130)
