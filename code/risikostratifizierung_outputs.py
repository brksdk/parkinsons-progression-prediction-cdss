# -*- coding: utf-8 -*-
"""
Risikostratifizierung-Outputs für Kapitel 4
===========================================

Dieses Skript erzeugt thesis-taugliche Tabellen und Abbildungen zur
biomarkerbasierten Risikostratifizierung des Parkinson-CDSS.

Es verändert das Streamlit-Dashboard NICHT. Es liest lediglich die bereits
exportierten Dashboard-Assets aus `dashboard_assets_full/` und speichert
separate Ergebnisse in einem Output-Ordner.

Erwartete Dateien im Asset-Ordner:
- xgb_model_full.pkl
- all_features.pkl
- test_data_full.pkl
- optional: peptide_map.pkl

Ausführung im Projektordner:
    python risikostratifizierung_outputs.py

Optionale Parameter:
    python risikostratifizierung_outputs.py --asset-dir dashboard_assets_full --output-dir thesis_outputs_risikostratifizierung
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


TARGET_COLS = [
    "updrs_1_plus_0",  "updrs_2_plus_0",  "updrs_3_plus_0",  "updrs_4_plus_0",
    "updrs_1_plus_6",  "updrs_2_plus_6",  "updrs_3_plus_6",  "updrs_4_plus_6",
    "updrs_1_plus_12", "updrs_2_plus_12", "updrs_3_plus_12", "updrs_4_plus_12",
    "updrs_1_plus_24", "updrs_2_plus_24", "updrs_3_plus_24", "updrs_4_plus_24",
]

CLINICAL_KEYWORDS = ["lag", "trend", "roll", "visit_month", "medication"]


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------

def ist_biomarker(feature_name: str) -> bool:
    """Gibt True zurück, wenn ein Merkmal als Protein-/Peptidmerkmal gilt."""
    name = str(feature_name).lower()
    return not any(keyword in name for keyword in CLINICAL_KEYWORDS)


def lade_peptid_mapping(asset_dir: Path) -> dict[str, str]:
    """Lädt optional eine Peptid-zu-UniProt-Zuordnung."""
    path = asset_dir / "peptide_map.pkl"
    if path.exists():
        try:
            mapping = joblib.load(path)
            if isinstance(mapping, dict):
                return mapping
        except Exception:
            pass
    return {}


def format_biomarker(feature_name: str, peptide_map: dict[str, str]) -> str:
    """Ergänzt bei Peptiden optional die UniProt-Information."""
    if feature_name in peptide_map:
        return f"{feature_name} ({peptide_map[feature_name]})"
    return feature_name


def waehle_top_biomarker(feature_names: list[str], importances: np.ndarray, top_n: int) -> list[str]:
    """Wählt die wichtigsten biologischen Merkmale anhand der Modellwichtigkeit aus."""
    imp_df = pd.DataFrame({"Merkmal": feature_names, "Modellwichtigkeit": importances})
    bio_features = [feature for feature in feature_names if ist_biomarker(feature)]
    top_bio = (
        imp_df[imp_df["Merkmal"].isin(bio_features)]
        .sort_values("Modellwichtigkeit", ascending=False)
        .head(top_n)["Merkmal"]
        .tolist()
    )
    if len(top_bio) < 2:
        raise ValueError(
            "Es wurden weniger als zwei biologische Merkmale gefunden. "
            "Bitte prüfen, ob die Feature-Namen und CLINICAL_KEYWORDS korrekt sind."
        )
    return top_bio


def risikoprofil_name(cluster_id: int, high_cluster: int) -> str:
    if int(cluster_id) == int(high_cluster):
        return "Höheres modellbasiertes Progressionsrisiko"
    return "Niedrigeres modellbasiertes Progressionsrisiko"


def lade_assets(asset_dir: Path) -> tuple[Any, list[str], pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Lädt Modell, Feature-Liste und Testdaten aus den Dashboard-Assets."""
    required_files = ["xgb_model_full.pkl", "all_features.pkl", "test_data_full.pkl"]
    missing = [name for name in required_files if not (asset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Folgende Asset-Dateien fehlen im Ordner "
            f"'{asset_dir}': {', '.join(missing)}"
        )

    model = joblib.load(asset_dir / "xgb_model_full.pkl")
    all_features = list(joblib.load(asset_dir / "all_features.pkl"))
    test_data = joblib.load(asset_dir / "test_data_full.pkl")

    if "X_test" not in test_data or "patient_ids" not in test_data:
        raise ValueError("test_data_full.pkl muss mindestens 'X_test' und 'patient_ids' enthalten.")

    X_test = test_data["X_test"]
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=all_features)

    X_test = X_test.reindex(columns=all_features)
    patient_ids = np.asarray(test_data["patient_ids"])

    if len(X_test) != len(patient_ids):
        raise ValueError(
            "Asset-Mismatch: X_test und patient_ids haben unterschiedliche Längen. "
            "Bitte Dashboard-Assets im Notebook erneut exportieren."
        )

    return model, all_features, X_test, patient_ids, test_data


def extrahiere_beobachteten_updrs3_24(test_data: dict[str, Any], n_rows: int) -> np.ndarray | None:
    """
    Versucht, beobachtete UPDRS-III-24-Werte aus optionalen Testdaten zu extrahieren.
    Falls nicht vorhanden, wird None zurückgegeben.
    """
    candidates = []
    for key in ["y_test", "y_true", "true_values", "predictions"]:
        if key in test_data:
            candidates.append(test_data[key])

    possible_cols = [
        "updrs_3_plus_24",
        "True_updrs_3_plus_24",
        "Wahr_updrs_3_plus_24",
        "Ist_updrs_3_plus_24",
    ]

    for obj in candidates:
        try:
            df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
            if len(df) != n_rows:
                continue
            for col in possible_cols:
                if col in df.columns:
                    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
    return None


def speichere_tabelle_csv_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path | None = None, max_latex_rows: int | None = None) -> list[Path]:
    saved = []
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    saved.append(csv_path)
    if tex_path is not None:
        latex_df = df.head(max_latex_rows) if max_latex_rows is not None else df
        tex_path.write_text(latex_df.to_latex(index=False, escape=True), encoding="utf-8")
        saved.append(tex_path)
    return saved


# -----------------------------------------------------------------------------
# Kernanalyse
# -----------------------------------------------------------------------------

def berechne_risikostratifizierung(
    model: Any,
    feature_names: list[str],
    X_test: pd.DataFrame,
    patient_ids: np.ndarray,
    test_data: dict[str, Any],
    peptide_map: dict[str, str],
    top_n_biomarker: int = 50,
    n_cluster: int = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Berechnet alle Tabellen und Plot-Daten für Kapitel 4."""
    if n_cluster != 2:
        raise ValueError("Dieses Thesis-Skript ist methodisch auf k=2 Risikoprofile ausgelegt.")

    importances = np.asarray(model.feature_importances_, dtype=float)
    top_bio_features = waehle_top_biomarker(feature_names, importances, top_n=top_n_biomarker)

    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_test[top_bio_features])

    predictions = model.predict(X_test)
    target_idx = TARGET_COLS.index("updrs_3_plus_24")
    pred_updrs3_24 = np.asarray(predictions[:, target_idx], dtype=float)

    cluster_means_pred = {
        cluster_id: float(np.mean(pred_updrs3_24[cluster_labels == cluster_id]))
        for cluster_id in sorted(np.unique(cluster_labels))
    }
    high_cluster = max(cluster_means_pred, key=cluster_means_pred.get)
    low_cluster = min(cluster_means_pred, key=cluster_means_pred.get)

    true_updrs3_24 = extrahiere_beobachteten_updrs3_24(test_data, len(X_test))

    observation_df = pd.DataFrame({
        "Patienten-ID": patient_ids,
        "Cluster": cluster_labels.astype(int),
        "Risikoprofil": [risikoprofil_name(c, high_cluster) for c in cluster_labels],
        "Prognostizierter UPDRS-III-Score nach 24 Monaten": np.round(pred_updrs3_24, 3),
    })
    if true_updrs3_24 is not None:
        observation_df["Beobachteter UPDRS-III-Score nach 24 Monaten"] = np.round(true_updrs3_24, 3)
        observation_df["Vorhersagefehler UPDRS-III nach 24 Monaten"] = np.round(
            observation_df["Prognostizierter UPDRS-III-Score nach 24 Monaten"] - observation_df["Beobachteter UPDRS-III-Score nach 24 Monaten"],
            3,
        )

    total_obs = len(observation_df)
    total_patients = observation_df["Patienten-ID"].nunique()
    summary_rows = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = observation_df["Cluster"] == cluster_id
        row = {
            "Cluster": int(cluster_id),
            "Risikoprofil": risikoprofil_name(cluster_id, high_cluster),
            "Anzahl Beobachtungen": int(mask.sum()),
            "Anteil Beobachtungen (%)": round(float(mask.mean() * 100), 2),
            "Anzahl Patienten": int(observation_df.loc[mask, "Patienten-ID"].nunique()),
            "Anteil Patienten (%)": round(float(observation_df.loc[mask, "Patienten-ID"].nunique() / total_patients * 100), 2),
            "Durchschnittlicher prognostizierter UPDRS-III-24-Monats-Score": round(float(observation_df.loc[mask, "Prognostizierter UPDRS-III-Score nach 24 Monaten"].mean()), 3),
            "Median prognostizierter UPDRS-III-24-Monats-Score": round(float(observation_df.loc[mask, "Prognostizierter UPDRS-III-Score nach 24 Monaten"].median()), 3),
            "Standardabweichung prognostizierter UPDRS-III-24-Monats-Score": round(float(observation_df.loc[mask, "Prognostizierter UPDRS-III-Score nach 24 Monaten"].std()), 3),
        }
        if true_updrs3_24 is not None:
            row["Durchschnittlicher beobachteter UPDRS-III-24-Monats-Score"] = round(float(observation_df.loc[mask, "Beobachteter UPDRS-III-Score nach 24 Monaten"].mean()), 3)
            row["Mittlerer Vorhersagefehler UPDRS-III nach 24 Monaten"] = round(float(observation_df.loc[mask, "Vorhersagefehler UPDRS-III nach 24 Monaten"].mean()), 3)
        summary_rows.append(row)
    cluster_summary_df = pd.DataFrame(summary_rows)

    cluster_means = X_test[top_bio_features].groupby(cluster_labels).mean()
    imp_lookup = dict(zip(feature_names, importances))
    biomarker_rows = []
    for feature in top_bio_features:
        high_mean = float(cluster_means.loc[high_cluster, feature])
        low_mean = float(cluster_means.loc[low_cluster, feature])
        diff = high_mean - low_mean
        biomarker_rows.append({
            "Biomarker": format_biomarker(feature, peptide_map),
            "Originales Merkmal": feature,
            "Mittelwert im Hochrisiko-Profil": round(high_mean, 4),
            "Mittelwert im Niedrigrisiko-Profil": round(low_mean, 4),
            "Differenz Hochrisiko minus Niedrigrisiko": round(diff, 4),
            "Absolute Differenz": round(abs(diff), 4),
            "Richtung": "höher im Hochrisiko-Profil" if diff > 0 else "niedriger im Hochrisiko-Profil",
            "Modellbasierte Wichtigkeit": round(float(imp_lookup.get(feature, np.nan)), 8),
        })
    biomarker_diff_df = pd.DataFrame(biomarker_rows).sort_values("Absolute Differenz", ascending=False).reset_index(drop=True)

    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(X_test[top_bio_features])
    pca_df = pd.DataFrame({
        "Hauptkomponente 1": pca_values[:, 0],
        "Hauptkomponente 2": pca_values[:, 1],
        "Cluster": cluster_labels.astype(int),
        "Risikoprofil": [risikoprofil_name(c, high_cluster) for c in cluster_labels],
        "Patienten-ID": patient_ids,
        "Prognostizierter UPDRS-III-Score nach 24 Monaten": np.round(pred_updrs3_24, 3),
    })

    try:
        sil_score = float(silhouette_score(X_test[top_bio_features], cluster_labels))
    except Exception:
        sil_score = np.nan

    validation_df = pd.DataFrame({
        "Metrik": [
            "Anzahl verwendeter Biomarker",
            "Anzahl Cluster",
            "Silhouette Score",
            "Erklärte Varianz PCA-Komponente 1 (%)",
            "Erklärte Varianz PCA-Komponente 2 (%)",
            "Hochrisiko-Cluster",
            "Niedrigrisiko-Cluster",
            "Beobachtete Zielwerte verfügbar",
        ],
        "Wert": [
            len(top_bio_features),
            n_cluster,
            round(sil_score, 4) if not np.isnan(sil_score) else "nicht berechenbar",
            round(float(pca.explained_variance_ratio_[0] * 100), 2),
            round(float(pca.explained_variance_ratio_[1] * 100), 2),
            int(high_cluster),
            int(low_cluster),
            "ja" if true_updrs3_24 is not None else "nein",
        ],
    })

    cdss_relevance_df = pd.DataFrame({
        "Ergebnisbaustein": [
            "K-Means-Cluster",
            "UPDRS-III-24-Monats-Vergleich",
            "Top-Biomarker-Differenzen",
            "PCA-Projektion",
            "Silhouette Score",
        ],
        "Bedeutung für das klinische Entscheidungsunterstützungssystem": [
            "Zuordnung eines Patienten zu einem biomarkerbasierten Risikoprofil",
            "Benennung der Cluster als höheres oder niedrigeres modellbasiertes Progressionsrisiko",
            "Anzeige clusterprägender biologischer Merkmale im Patientenprofil",
            "Explorative Visualisierung der Trennbarkeit im biologischen Merkmalsraum",
            "Vorsichtige Einordnung der Clusterqualität; keine klinische Subtypdiagnose",
        ],
    })

    return {
        "top_bio_features": top_bio_features,
        "cluster_labels": cluster_labels,
        "high_cluster": high_cluster,
        "low_cluster": low_cluster,
        "observation_df": observation_df,
        "cluster_summary_df": cluster_summary_df,
        "biomarker_diff_df": biomarker_diff_df,
        "pca_df": pca_df,
        "validation_df": validation_df,
        "cdss_relevance_df": cdss_relevance_df,
    }


# -----------------------------------------------------------------------------
# Speichern von Tabellen und Abbildungen
# -----------------------------------------------------------------------------

def speichere_outputs(results: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    saved += speichere_tabelle_csv_tex(
        results["cluster_summary_df"],
        output_dir / "cluster_zusammenfassung.csv",
        output_dir / "tab_cluster_zusammenfassung.tex",
    )
    saved += speichere_tabelle_csv_tex(
        results["observation_df"],
        output_dir / "cluster_beobachtungen.csv",
    )
    saved += speichere_tabelle_csv_tex(
        results["biomarker_diff_df"],
        output_dir / "top_biomarker_clustervergleich.csv",
        output_dir / "tab_top10_biomarker_clustervergleich.tex",
        max_latex_rows=10,
    )
    saved += speichere_tabelle_csv_tex(
        results["pca_df"],
        output_dir / "pca_biomarker_cluster_daten.csv",
    )
    saved += speichere_tabelle_csv_tex(
        results["validation_df"],
        output_dir / "cluster_validierung_metriken.csv",
        output_dir / "tab_cluster_validierung.tex",
    )
    saved += speichere_tabelle_csv_tex(
        results["cdss_relevance_df"],
        output_dir / "cdss_relevanz_tabelle.csv",
        output_dir / "tab_cdss_relevanz.tex",
    )

    # Excel-Sammeldatei, falls openpyxl installiert ist.
    try:
        xlsx_path = output_dir / "risikostratifizierung_ergebnisse.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            results["cluster_summary_df"].to_excel(writer, sheet_name="Cluster-Zusammenfassung", index=False)
            results["observation_df"].to_excel(writer, sheet_name="Beobachtungen", index=False)
            results["biomarker_diff_df"].to_excel(writer, sheet_name="Top-Biomarker", index=False)
            results["validation_df"].to_excel(writer, sheet_name="Validierung", index=False)
            results["cdss_relevance_df"].to_excel(writer, sheet_name="CDSS-Relevanz", index=False)
        saved.append(xlsx_path)
    except Exception:
        pass

    # Abbildung 1: Boxplot der Prognose nach Risikoprofil
    box_path = output_dir / "boxplot_updrs3_24_nach_risikoprofil.png"
    plot_df = results["observation_df"].copy()
    profiles = sorted(plot_df["Risikoprofil"].unique().tolist())
    data = [plot_df.loc[plot_df["Risikoprofil"] == profile, "Prognostizierter UPDRS-III-Score nach 24 Monaten"].values for profile in profiles]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data, labels=profiles, showmeans=True)
    ax.set_title("Prognostizierter UPDRS-III-24-Monats-Score nach Risikoprofil")
    ax.set_xlabel("Biomarkerbasiertes Risikoprofil")
    ax.set_ylabel("Prognostizierter UPDRS-III-Score nach 24 Monaten")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(box_path)

    # Abbildung 2: Top-10 Biomarker-Differenzen
    biomarker_path = output_dir / "balkendiagramm_top10_biomarker_clustervergleich.png"
    top10 = results["biomarker_diff_df"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top10["Biomarker"], top10["Absolute Differenz"])
    ax.set_title("Top-10 Biomarker mit größter Differenz zwischen den Risikoprofilen")
    ax.set_xlabel("Absolute Differenz der standardisierten Mittelwerte")
    ax.set_ylabel("Biomarker")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(biomarker_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(biomarker_path)

    # Abbildung 3: PCA-Projektion
    pca_path = output_dir / "pca_biomarker_cluster.png"
    pca_df = results["pca_df"].copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    for profile in pca_df["Risikoprofil"].unique():
        subset = pca_df[pca_df["Risikoprofil"] == profile]
        ax.scatter(subset["Hauptkomponente 1"], subset["Hauptkomponente 2"], label=profile, alpha=0.75)
    ax.set_title("PCA-Projektion der biomarkerbasierten Risikoprofile")
    ax.set_xlabel("Hauptkomponente 1")
    ax.set_ylabel("Hauptkomponente 2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(pca_path)

    # Kurzer maschinenlesbarer Analyse-Report
    report_path = output_dir / "analyse_report.json"
    report = {
        "hochrisiko_cluster": int(results["high_cluster"]),
        "niedrigrisiko_cluster": int(results["low_cluster"]),
        "anzahl_biomarker": int(len(results["top_bio_features"])),
        "erzeugte_dateien": [path.name for path in saved],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    saved.append(report_path)

    zip_path = output_dir / "risikostratifizierung_ergebnisse.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in saved:
            if path.exists():
                zip_file.write(path, arcname=path.name)
    saved.append(zip_path)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erzeugt Kapitel-4-Outputs zur biomarkerbasierten Risikostratifizierung.")
    parser.add_argument("--asset-dir", type=Path, default=Path("dashboard_assets_full"), help="Ordner mit den Dashboard-Assets.")
    parser.add_argument("--output-dir", type=Path, default=Path("thesis_outputs_risikostratifizierung"), help="Zielordner für Tabellen und Abbildungen.")
    parser.add_argument("--top-n-biomarker", type=int, default=50, help="Anzahl der biologischen Merkmale für K-Means.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed für K-Means und reproduzierbare Ergebnisse.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("--- Biomarkerbasierte Risikostratifizierung: Thesis-Output ---")
    print(f"Asset-Ordner: {args.asset_dir}")
    print(f"Output-Ordner: {args.output_dir}")

    model, all_features, X_test, patient_ids, test_data = lade_assets(args.asset_dir)
    peptide_map = lade_peptid_mapping(args.asset_dir)

    results = berechne_risikostratifizierung(
        model=model,
        feature_names=all_features,
        X_test=X_test,
        patient_ids=patient_ids,
        test_data=test_data,
        peptide_map=peptide_map,
        top_n_biomarker=args.top_n_biomarker,
        n_cluster=2,
        random_state=args.random_state,
    )

    saved_files = speichere_outputs(results, args.output_dir)

    print("\nCluster-Zusammenfassung:")
    print(results["cluster_summary_df"].to_string(index=False))

    print("\nValidierungsmetriken:")
    print(results["validation_df"].to_string(index=False))

    print("\nTop-10 Biomarker nach Cluster-Differenz:")
    print(results["biomarker_diff_df"].head(10).to_string(index=False))

    print("\nGespeicherte Dateien:")
    for path in saved_files:
        print(f"- {path}")
if __name__ == "__main__":
    main()
