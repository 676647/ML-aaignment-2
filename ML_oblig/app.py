#!/usr/bin/env python3
# app.py — Music Genre Classifier (dataset-only, no external APIs)

import os
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from typing import List, Tuple

# -----------------------------
# Konfigurasjon
# -----------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "./GenreDetectorV2/models")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABELENC_PATH = os.path.join(MODEL_DIR, "labelencoder.joblib")

# Sti til Kaggle CSV (kan overstyres med env)
DATASET_CSV = os.getenv("DATASET_CSV", "./GenreDetectorV2/data/train.csv")

# Funksjonssett som modellen forventer (må matche treningen)
FEATURE_COLUMNS = [
    "duration_ms", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature"
]

# -----------------------------
# Last model artefakter
# -----------------------------
def _load_artifacts():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, LABELENC_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Finner ikke modellfiler. Mangler:\n" + "\n".join(missing) +
            "\n\nKjør treningsscriptet så filene ligger i ./GenreDetectorV2/models/."
        )
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABELENC_PATH)
    return clf, scaler, le

clf, scaler, le = _load_artifacts()

def _validate_pipeline():
    problems = []
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(FEATURE_COLUMNS):
        problems.append(f"Scaler forventer {scaler.n_features_in_} features, men FEATURE_COLUMNS har {len(FEATURE_COLUMNS)}.")
    if hasattr(clf, "n_features_in_") and clf.n_features_in_ != len(FEATURE_COLUMNS):
        problems.append(f"Modellen forventer {clf.n_features_in_} features, men FEATURE_COLUMNS har {len(FEATURE_COLUMNS)}.")
    if hasattr(scaler, "feature_names_in_"):
        if list(scaler.feature_names_in_) != FEATURE_COLUMNS:
            problems.append(
                "Kolonnenavn/rekkefølge mismatch mellom scaler og FEATURE_COLUMNS.\n"
                f"Scaler:   {list(scaler.feature_names_in_)}\n"
                f"Forventet:{FEATURE_COLUMNS}"
            )
    if problems:
        print("⚠️ Pipelinevalidering:")
        for p in problems: print(" -", p)

_validate_pipeline()

# -----------------------------
# Last datasett i minne
# -----------------------------
df_cache: pd.DataFrame | None = None
id_to_row = {}
search_index: list[tuple[str, str]] = []  # (visningslabel, track_id)

def _safe_str(x):
    return str(x) if pd.notna(x) else ""

def load_local_dataset():
    global df_cache, id_to_row, search_index
    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(
            f"Fant ikke datasett CSV på: {DATASET_CSV}\n"
            "Sett miljøvariabel DATASET_CSV eller flytt train.csv til standardstien."
        )

    df = pd.read_csv(DATASET_CSV)

    # Normaliser kolonnenavn hvis 'track_genre' brukes i rå CSV
    if "track_genre" in df.columns and "genre" not in df.columns:
        df = df.rename(columns={"track_genre": "genre"})

    # Behold relevante kolonner
    needed = {"track_id", "track_name", "artists", *FEATURE_COLUMNS}
    keep = [c for c in df.columns if c in needed]
    missing_feats = [c for c in FEATURE_COLUMNS if c not in keep]
    if missing_feats:
        # Ikke kritisk for kjøring, men greit å vite
        print(f"⚠️ Mangler kolonner i CSV: {missing_feats}. Sørg for at trenings-CSV matcher FEATURE_COLUMNS.")

    df = df[keep].copy()

    # Bygg søketekst (for enkel substring-søk)
    name_col = "track_name" if "track_name" in df.columns else None
    artist_col = "artists" if "artists" in df.columns else None
    df["__search_text"] = (
        (df[name_col].astype(str) if name_col else "") + " " +
        (df[artist_col].astype(str) if artist_col else "")
    ).str.lower()

    # Cache
    df_cache = df
    id_to_row = {}
    search_index = []
    for _, r in df_cache.iterrows():
        tid = r.get("track_id")
        if pd.isna(tid):
            continue
        id_to_row[tid] = r
        label = f"{_safe_str(r.get('track_name'))} — {_safe_str(r.get('artists'))}"
        search_index.append((label, tid))

load_local_dataset()

# -----------------------------
# Prediksjon
# -----------------------------
def predict_from_features(df: pd.DataFrame) -> tuple[str, list[tuple[str, float]]]:
    # korrekt rekkefølge + type-coerce
    df = df[FEATURE_COLUMNS].copy()

    int_cols = ["key", "mode", "time_signature", "duration_ms"]
    float_cols = [c for c in FEATURE_COLUMNS if c not in int_cols]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != df.shape[1]:
        raise ValueError(f"Scaler forventer {scaler.n_features_in_} features, fikk {df.shape[1]}.")
    if hasattr(clf, "n_features_in_") and clf.n_features_in_ != df.shape[1]:
        raise ValueError(f"Modellen forventer {clf.n_features_in_} features, fikk {df.shape[1]}.")

    X_sc = scaler.transform(df)
    y_pred_enc = clf.predict(X_sc)[0]
    pred_label = le.inverse_transform([y_pred_enc])[0]

    top5 = []
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_sc)[0]
        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [(le.inverse_transform([i])[0], float(probs[i])) for i in top5_idx]
    return pred_label, top5

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="🎧 Genre Classifier (dataset-only)") as demo:
    gr.Markdown(
        "# 🎶 Music Genre Classification — Datasett\n"
        "Søk i **Kaggle-datasettet** lokalt og klassifiser sjanger med en trenet modell.\n"
        "Ingen eksterne APIer benyttes."
    )

    with gr.Tab("Datasett-søk"):
        ds_query = gr.Textbox(label="Søk (tittel / artist)", placeholder="f.eks. queen bohemian")
        ds_search_btn = gr.Button("Søk i datasett")

        ds_results = gr.Dropdown(label="Velg spor", choices=[], interactive=True)
        ds_info = gr.Markdown("")
        ds_predict_btn = gr.Button("Klassifiser valgt låt")

        ds_pred_label = gr.Label(label="Predikert sjanger")
        ds_top5 = gr.JSON(label="Top-5 sannsynligheter")

        ds_cache = gr.State([])  # [{'label':..., 'id':...}, ...]

        def ds_do_search(q):
            if df_cache is None or len(search_index) == 0:
                return gr.update(choices=[]), [], "Ingen datasett lastet (sjekk DATASET_CSV)."
            if not q or not q.strip():
                return gr.update(choices=[]), [], "Skriv noe for å søke."

            ql = q.strip().lower()
            # substring-søk på precomputet search_text eller label
            matches = []
            for (label, tid) in search_index:
                # rask sjekk i label først
                if ql in label.lower():
                    matches.append((label, tid))
                else:
                    # fallback: søk i __search_text (tittel+artist)
                    row = id_to_row.get(tid)
                    if row is not None:
                        st = row.get("__search_text", "")
                        if isinstance(st, str) and ql in st:
                            matches.append((label, tid))

            matches = matches[:100]  # cap for UI
            if not matches:
                return gr.update(choices=[]), [], "Ingen treff."
            choices = [m[0] for m in matches]
            cache = [{"label": m[0], "id": m[1]} for m in matches]
            return gr.update(choices=choices, value=(choices[0] if choices else None)), cache, "Velg et spor."

        def ds_show(selected, cache):
            try:
                if not selected:
                    return "Ingen valgt."
                item = next((c for c in cache if c["label"] == selected), None)
                if not item:
                    return "Fant ikke valgt spor."
                row = id_to_row.get(item["id"])
                if row is None:
                    return "Fant ikke raddata for dette track_id."
                title = row.get("track_name", "")
                artist = row.get("artists", "")
                dur = row.get("duration_ms", "")
                # valgfritt: vis enkelte featureverdier
                return (
                    f"**{title}** — {artist}\n\n"
                    f"Varighet: {dur} ms\n\n"
                    f"`track_id`: `{item['id']}`"
                )
            except Exception as e:
                return f"❌ Feil ved visning: {type(e).__name__}: {e}"

        def ds_predict(selected, cache):
            try:
                if not selected:
                    return "Ingen valgt", []
                item = next((c for c in cache if c["label"] == selected), None)
                if not item:
                    return "Fant ikke valgt spor", []
                row = id_to_row.get(item["id"])
                if row is None:
                    return "Fant ikke raddata for valgt spor", []

                # bygg DF med riktige kolonner
                feat_row = {k: row[k] for k in FEATURE_COLUMNS if k in row.index}
                df = pd.DataFrame([feat_row], columns=FEATURE_COLUMNS)

                pred, top5 = predict_from_features(df)
                return pred, top5
            except Exception as e:
                return f"❌ Feil under prediksjon: {type(e).__name__}: {e}", []

        ds_search_btn.click(ds_do_search, [ds_query], [ds_results, ds_cache, ds_info])
        ds_results.change(ds_show, [ds_results, ds_cache], [ds_info])
        ds_predict_btn.click(ds_predict, [ds_results, ds_cache], [ds_pred_label, ds_top5])

    gr.Markdown(
        "ℹ️ Sørg for at `DATASET_CSV` peker til `train.csv` (f.eks. `./GenreDetectorV2/data/train.csv`).\n\n"
        "Modellfiler må ligge i `./GenreDetectorV2/models/`."
    )

# Start server — velg automatisk ledig port og vis feil i UI
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=None, show_error=True, share=True)
