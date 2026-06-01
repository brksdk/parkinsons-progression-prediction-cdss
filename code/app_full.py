import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Parkinson CDSS | Full-Feature Modell",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0d6efd; color: white; border-radius: 8px; font-weight: bold;}
    .metric-card {background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif;}
    .alert-card { border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; border-left: 4px solid; }
    .alert-high { background: #fef2f2; border-color: #ef4444; color: #7f1d1d; }
    .alert-mid  { background: #fffbeb; border-color: #f59e0b; color: #78350f; }
    .alert-low  { background: #f0fdf4; border-color: #22c55e; color: #14532d; }
    .small-note {font-size: 0.88rem; color: #6b7280;}

    /* Tab-Beschriftungen etwas größer als der Streamlit-Standard (~1rem) */
    button[data-baseweb="tab"] p {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
    }
    
    /* Subheader leicht verkleinert – wirkt sonst zu wuchtig neben den Tab-Titeln */
    .stMarkdown h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
ASSET_DIR = Path("dashboard_assets_full")
UPDRS3_RISK_THRESHOLD = 35.0
UPDRS3_MODERATE_THRESHOLD = 25.0

TARGET_COLS = [
    'updrs_1_plus_0',  'updrs_2_plus_0',  'updrs_3_plus_0',  'updrs_4_plus_0',
    'updrs_1_plus_6',  'updrs_2_plus_6',  'updrs_3_plus_6',  'updrs_4_plus_6',
    'updrs_1_plus_12', 'updrs_2_plus_12', 'updrs_3_plus_12', 'updrs_4_plus_12',
    'updrs_1_plus_24', 'updrs_2_plus_24', 'updrs_3_plus_24', 'updrs_4_plus_24'
]

CLINICAL_KEYWORDS = ['lag', 'trend', 'roll', 'visit_month', 'medication']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def is_bio_feature(feature_name: str) -> bool:
    """Returns True for protein/peptide features and False for engineered clinical features."""
    return not any(keyword in feature_name.lower() for keyword in CLINICAL_KEYWORDS)


def classify_updrs3_risk(score: float) -> str:
    """Risk class used for dashboard-level monitoring, not for direct diagnosis."""
    if score >= UPDRS3_RISK_THRESHOLD:
        return "🟥 Hohes modellbasiertes Risiko"
    if score >= UPDRS3_MODERATE_THRESHOLD:
        return "🟨 Mittleres modellbasiertes Risiko"
    return "🟩 Niedriges modellbasiertes Risiko"


def estimate_threshold_crossing(months, values, threshold=UPDRS3_RISK_THRESHOLD):
    """Linear interpolation of the first threshold crossing for UPDRS-3."""
    if values[0] >= threshold:
        return "Bereits im Ausgangszeitpunkt im erhöhten Risikobereich", "#ef4444"

    for i in range(len(months) - 1):
        if values[i] < threshold <= values[i + 1]:
            slope = (values[i + 1] - values[i]) / (months[i + 1] - months[i])
            if slope == 0:
                return "Unklar", "#f59e0b"
            exact_month = months[i] + (threshold - values[i]) / slope
            return f"Voraussichtlich Monat {exact_month:.1f}", "#ef4444"

    return "> 24 Monate / keine Überschreitung im Prognosefenster", "#22c55e"


def get_target_mae(target_name: str, model_name: str = "XGBoost"):
    metrics_file = [
        ASSET_DIR / "finale_metriken.csv",
        Path("finale_metriken.csv"),
    ]

    for file in metrics_file:
        if not file.exists():
            continue
        try:
            df = pd.read_csv(file)
            model_col = "Modell" if "Modell" in df.columns else "Model" if "Model" in df.columns else None
            target_col = "Zielvariable" if "Zielvariable" in df.columns else "Target" if "Target" in df.columns else None
            if model_col is None or target_col is None or "MAE" not in df.columns:
                continue

            row = df[(df[model_col].astype(str) == model_name) & (df[target_col].astype(str) == target_name)]
            if not row.empty:
                return float(row.iloc[0]["MAE"])
        except Exception:
            continue
    return None


# =============================================================================
# LOAD ASSETS
# =============================================================================
@st.cache_resource(show_spinner="Daten werden geladen und verifiziert...")
def load_assets():
    model = joblib.load(ASSET_DIR / 'xgb_model_full.pkl')
    y_scaler = joblib.load(ASSET_DIR / 'y_scaler.pkl')
    all_features = joblib.load(ASSET_DIR / 'all_features.pkl')
    test_data = joblib.load(ASSET_DIR / 'test_data_full.pkl')

    required_keys = ["X_test", "patient_ids"]
    missing = [key for key in required_keys if key not in test_data]
    if missing:
        raise ValueError(f"test_data_full.pkl enthält nicht alle erforderlichen Schlüssel: {missing}")

    # Zeilen dürfen hier nicht stillschweigend wegfallen – sonst läuft die ID-Zuordnung aus dem Takt.
    if len(test_data['X_test']) != len(test_data['patient_ids']):
        raise ValueError(
            "Asset mismatch: X_test und patient_ids haben unterschiedliche Längen. "
            "Bitte die Dashboard-Assets im Notebook erneut exportieren, ohne Zeilen zu kürzen."
        )

    if list(test_data['X_test'].columns) != list(all_features):
        # Nicht zwingend ein Abbruchfehler, aber in klinischen Dashboards muss die Feature-Reihenfolge eindeutig sein.
        test_data['X_test'] = test_data['X_test'].reindex(columns=all_features)

    return model, y_scaler, all_features, test_data


@st.cache_data(show_spinner=False)
def load_peptide_map():
    try:
        return joblib.load(ASSET_DIR / 'peptide_map.pkl')
    except FileNotFoundError:
        return {}


try:
    xgb_model, y_scaler, all_features, test_data = load_assets()
except Exception as exc:
    st.error(f"Fehler beim Laden der Dashboard-Assets: {exc}")
    st.stop()

peptide_map = load_peptide_map()


def fmt(feature_name: str) -> str:
    """Adds UniProt information to peptide names when available."""
    if feature_name in peptide_map:
        return f"{feature_name} ({peptide_map[feature_name]})"
    return feature_name


X_test_full = test_data['X_test']
patient_ids = np.asarray(test_data['patient_ids'])
all_predictions_df = test_data.get('predictions', None)

# Aliase für ältere Dashboard-Varianten – damit muss ich den Rest des Codes nicht anfassen.
selected_features = all_features
X_test_opt = X_test_full

# =============================================================================
# CACHED COMPUTATIONS
# =============================================================================
@st.cache_data(show_spinner=False)
def compute_clusters(_X, bio_cols_tuple):
    bio_cols = list(bio_cols_tuple)
    if len(bio_cols) < 2:
        return np.zeros(len(_X), dtype=int)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    return kmeans.fit_predict(_X[bio_cols])


@st.cache_data(show_spinner=False)
def get_top_bio_features(_features_tuple, _importances_tuple, top_n=50):
    features = list(_features_tuple)
    importances = list(_importances_tuple)
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    bio_features = [feature for feature in features if is_bio_feature(feature)]
    return (
        imp_df[imp_df['Feature'].isin(bio_features)]
        .sort_values('Importance', ascending=False)
        .head(top_n)['Feature']
        .tolist()
    )


# =============================================================================
# SESSION STATE
# =============================================================================
_defaults = {
    "messages": [],
    "shap_context": "Noch keine SHAP-Analyse durchgeführt.",
    "pred_context": None,
    "current_prediction": None,
    "sim_prediction": None,
    "crossing_month": "Unbekannt / keine Vorhersage generiert",
    "subtype_name": "Nicht analysiert",
    "risk_level": "Nicht berechnet",
    "last_patient": None,
}
for key, default_value in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# =============================================================================
# GLOBAL APP HEADER + DISCLAIMER
# =============================================================================
st.title("🔬 Parkinson CDSS")
st.caption(
    "Personalisierte Fortschrittsvorhersage mit klinischen Daten, Proteomik, Peptidomik, "
    "maschinellem Lernen und erklärbarer KI."
)


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🧠 Parkinson CDSS")
st.sidebar.markdown("**Modellbasierte klinische Entscheidungsunterstützung**")

unique_patients = np.unique(patient_ids)
selected_patient = st.sidebar.selectbox("Patienten-ID auswählen", unique_patients)

patient_indices = np.where(patient_ids == selected_patient)[0]
if len(patient_indices) == 0:
    st.error("Die ausgewählte Patienten-ID wurde nicht im Testdatensatz gefunden.")
    st.stop()

idx = int(patient_indices[0])
patient_data = X_test_full.iloc[[idx]]

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 What-if Behandlungssimulator")
sim_med_status = st.sidebar.radio(
    "Medikationsstatus simulieren:",
    ["Ursprünglicher Zustand", "Mit Medikamenten", "Ohne Medikamente"],
    index=0,
    help=(
        "Verändert ausschließlich den kodierten Medikationsstatus im Modellinput. "
    )
)

if st.session_state.last_patient != selected_patient:
    for key in ["current_prediction", "sim_prediction", "pred_context", "crossing_month", "subtype_name", "risk_level", "shap_context"]:
        st.session_state[key] = _defaults[key]
    st.session_state.last_patient = selected_patient

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Patientenprofil & Omik",
    "📈 Krankheitsverlauf",
    "🔬 Erklärbare KI (XAI)",
    "💊 Klinische Handlungshinweise",
    "📄 Bericht herunterladen",
    "ℹ️ Modellinformationen"
])

# =============================================================================
# TAB 1 — PATIENT PROFILE + BIOMARKER SUBTYPING
# =============================================================================
with tab1:
    st.subheader(f"👤 Patienten-ID: {selected_patient} – Klinisches & Multi-Omik Profil")

    bio_cols = get_top_bio_features(tuple(all_features), tuple(xgb_model.feature_importances_), top_n=50)

    with st.spinner("Biomarkerbasierter Risikotyp wird berechnet..."):
        cluster_labels = compute_clusters(X_test_full, tuple(bio_cols))
        patient_cluster = int(cluster_labels[idx])

        target_idx_updrs3_24 = TARGET_COLS.index('updrs_3_plus_24')
        c0_mask = cluster_labels == 0
        c1_mask = cluster_labels == 1
        c0_risk = xgb_model.predict(X_test_full[c0_mask])[:, target_idx_updrs3_24].mean() if c0_mask.any() else np.nan
        c1_risk = xgb_model.predict(X_test_full[c1_mask])[:, target_idx_updrs3_24].mean() if c1_mask.any() else np.nan

        high_cluster = 0 if c0_risk >= c1_risk else 1
        is_high = patient_cluster == high_cluster

        if is_high:
            subtype_name = "Biomarker-Subtyp A: höheres modellbasiertes Progressionsrisiko"
            short_subtype = "Subtyp A"
            other_short = "Subtyp B"
            subtype_color = "#ef4444"
            subtype_desc = (
                "Das Multi-Omik-Profil liegt in dem Cluster, dessen durchschnittliche UPDRS-3-24-Monats-Prognose "
                "höher ausfällt. Dies ist eine modellbasierte Risikostratifizierung, keine klinische Diagnose."
            )
        else:
            subtype_name = "Biomarker-Subtyp B | niedrigeres modellbasiertes Progressionsrisiko"
            short_subtype = "Subtyp B"
            other_short = "Subtyp A"
            subtype_color = "#22c55e"
            subtype_desc = (
                "Das Multi-Omik-Profil liegt in dem Cluster, dessen durchschnittliche UPDRS-3-24-Monats-Prognose "
                "niedriger ausfällt. Dies ist eine modellbasierte Risikostratifizierung, keine klinische Diagnose."
            )

        st.session_state.subtype_name = subtype_name

    st.markdown(f"""
    <div style='padding:15px; border-radius:10px; border: 2px solid {subtype_color}; background-color:{subtype_color}10;'>
        <h4 style='color:{subtype_color}; margin:0;'>Unüberwachte biomarkerbasierte Risikostratifizierung: {subtype_name}</h4>
        <p style='margin:0; font-size:14px;'>{subtype_desc}</p>
    </div><br>
    """, unsafe_allow_html=True)

    st.caption(
        "Wichtig: Die Cluster werden nicht als klinisch diagnostizierte PIGD-/TD-Subtypen interpretiert, "
        "sondern als datengetriebene Biomarker-Risikoprofile."
    )

    col_p1, col_p2 = st.columns([1, 1])

    cluster_means = X_test_full[bio_cols].groupby(cluster_labels).mean()
    diff_series = (cluster_means.loc[0] - cluster_means.loc[1]).abs().sort_values(ascending=False)

    with col_p1:
        st.markdown("**📈 Biomarker-Signaturen, die die Cluster-Zuordnung prägen**")
        st.caption("Gezeigt werden die Merkmale mit der größten Differenz zwischen beiden datengetriebenen Clustern.")

        top_diff_feats = diff_series.head(5).index.tolist()
        top_diff_labels = [fmt(feature) for feature in top_diff_feats]

        fig_sub = go.Figure()
        fig_sub.add_trace(go.Bar(
            name=f'{short_subtype} Cluster-Durchschnitt',
            x=top_diff_labels,
            y=cluster_means.loc[patient_cluster, top_diff_feats].values,
            marker_color=subtype_color,
            opacity=0.5
        ))
        fig_sub.add_trace(go.Bar(
            name=f'{other_short} Cluster-Durchschnitt',
            x=top_diff_labels,
            y=cluster_means.loc[1 - patient_cluster, top_diff_feats].values,
            marker_color='gray',
            opacity=0.3
        ))
        fig_sub.add_trace(go.Scatter(
            name='Aktueller Patientenwert',
            x=top_diff_labels,
            y=patient_data[top_diff_feats].iloc[0].values,
            mode='markers',
            marker=dict(color='black', size=12, symbol='diamond', line=dict(width=2, color='white'))
        ))
        fig_sub.update_layout(
            barmode='group',
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    with col_p2:
        st.markdown("**🕸️ Biologisches Profil: Patient vs. Population**")
        top_bio = diff_series.head(6).index.tolist()
        patient_bio_vals = patient_data[top_bio].iloc[0].values
        baseline_bio_vals = X_test_full[top_bio].mean().values
        top_bio_labels = [fmt(feature) for feature in top_bio]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_bio_vals,
            theta=top_bio_labels,
            fill='toself',
            name='Populationsdurchschnitt',
            line_color='gray'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=patient_bio_vals,
            theta=top_bio_labels,
            fill='toself',
            name=f'Patient {selected_patient}',
            line_color='#0d6efd'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# =============================================================================
# TAB 2 — PROGRESSION PREDICTION
# =============================================================================
with tab2:
    st.subheader("📈 Krankheitsverlauf und Time-to-Event-Analyse")

    if st.button("Vorhersage generieren & Analyse starten", type="primary", use_container_width=True):
        with st.spinner("Modell wird berechnet..."):
            prediction = xgb_model.predict(patient_data)[0]
            st.session_state.current_prediction = prediction

    if st.session_state.current_prediction is None:
        st.warning("Bitte zuerst eine Vorhersage generieren.")
    else:
        prediction = st.session_state.current_prediction

        # What-if-Simulation: nur die Medikations-Dummy-Features werden im skalierten Merkmalsraum verändert.
        if sim_med_status != "Ursprünglicher Zustand":
            sim_data = patient_data.copy()
            med_cols_sim = [col for col in sim_data.columns if 'medication' in col.lower()]

            if med_cols_sim:
                for col in med_cols_sim:
                    sim_data[col] = X_test_full[col].min()

                if "Mit Medikamenten" in sim_med_status:
                    on_col = [col for col in med_cols_sim if col.lower().endswith('_on')]
                    if on_col:
                        sim_data[on_col[0]] = X_test_full[on_col[0]].max()
                elif "Ohne Medikamente" in sim_med_status:
                    off_col = [col for col in med_cols_sim if col.lower().endswith('_off')]
                    if off_col:
                        sim_data[off_col[0]] = X_test_full[off_col[0]].max()

                st.session_state.sim_prediction = xgb_model.predict(sim_data)[0]
            else:
                st.session_state.sim_prediction = None
        else:
            st.session_state.sim_prediction = None

        sim_prediction = st.session_state.sim_prediction

        selected_updrs_part = st.selectbox(
            "Zu visualisierende Analysedimension auswählen",
            ["Teil 1 (Nicht-Motorisch)", "Teil 2 (Tägliches Leben)", "Teil 3 (Motorisch)", "Teil 4 (Komplikationen)"],
            index=2
        )
        part_num = {
            "Teil 1 (Nicht-Motorisch)": "1",
            "Teil 2 (Tägliches Leben)": "2",
            "Teil 3 (Motorisch)": "3",
            "Teil 4 (Komplikationen)": "4"
        }[selected_updrs_part]

        months = [0, 6, 12, 24]
        updrs_traj = [prediction[TARGET_COLS.index(f'updrs_{part_num}_plus_{month}')] for month in months]

        crossing_month = "Nur für UPDRS-3 berechnet"
        alert_color = "#6c757d"
        if part_num == "3":
            crossing_month, alert_color = estimate_threshold_crossing(months, updrs_traj, UPDRS3_RISK_THRESHOLD)
        st.session_state.crossing_month = crossing_month

        updrs3_24 = float(prediction[TARGET_COLS.index('updrs_3_plus_24')])
        risk_level = classify_updrs3_risk(updrs3_24)
        st.session_state.pred_context = updrs3_24
        st.session_state.risk_level = risk_level

        current_target = f'updrs_{part_num}_plus_0'
        future_target = f'updrs_{part_num}_plus_24'
        current_score = float(prediction[TARGET_COLS.index(current_target)])
        future_score = float(prediction[TARGET_COLS.index(future_target)])
        future_mae = get_target_mae(future_target, model_name="XGBoost")

        c1, c2, c3 = st.columns(3)
        c1.metric("Aktueller modellierter Score", f"{current_score:.1f}")
        if future_mae is not None:
            c2.metric(
                "24-Monats-Prognose",
                f"{future_score:.1f}",
                help=f"Durchschnittlicher Modellfehler für dieses Ziel: ±{future_mae:.1f} Punkte"
            )
            c2.caption(f"Interpretationsbereich: ca. {future_score - future_mae:.1f} bis {future_score + future_mae:.1f}")
        else:
            c2.metric("24-Monats-Prognose", f"{future_score:.1f}")
            c2.caption("Kein MAE-Asset gefunden. Für die finale Thesis-Version sollte ein Fehlerbereich ergänzt werden.")

        if part_num == "3":
            c3.markdown(
                f"<div style='text-align:center; padding:10px; border-radius:10px; background-color:{alert_color}20;'>"
                f"<h3 style='color:{alert_color}; margin:0;'>Zeit bis zum Risikomarker:<br>{crossing_month}</h3></div>",
                unsafe_allow_html=True
            )
        else:
            c3.metric("UPDRS-3-basierte Risikostufe", risk_level)

        st.caption(
            f"Der Schwellenwert von {UPDRS3_RISK_THRESHOLD:.0f} Punkten wird hier als heuristischer, "
            "modellbasierter Risikomarker verwendet und ist nicht als universeller klinischer Grenzwert zu verstehen."
        )

        st.divider()
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            pop_avg_traj = []
            if all_predictions_df is not None:
                for month in months:
                    true_col = f'True_updrs_{part_num}_plus_{month}'
                    pop_avg_traj.append(
                        all_predictions_df[true_col].mean() if true_col in all_predictions_df.columns else 0
                    )

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=months,
                y=updrs_traj,
                mode='lines+markers',
                name='Ursprüngliche Prognose',
                line=dict(color='#0d6efd', width=4)
            ))

            if sim_prediction is not None:
                sim_traj = [sim_prediction[TARGET_COLS.index(f'updrs_{part_num}_plus_{month}')] for month in months]
                fig_traj.add_trace(go.Scatter(
                    x=months,
                    y=sim_traj,
                    mode='lines+markers',
                    name=f'Szenario: {sim_med_status}',
                    line=dict(color='#ef4444', dash='dash', width=4)
                ))

            if pop_avg_traj and sum(pop_avg_traj) > 0:
                fig_traj.add_trace(go.Scatter(
                    x=months,
                    y=pop_avg_traj,
                    mode='lines',
                    name='Populationsdurchschnitt',
                    line=dict(color='#6c757d', dash='dot', width=2)
                ))

            if part_num == "3":
                fig_traj.add_hline(
                    y=UPDRS3_RISK_THRESHOLD,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Heuristischer Risikomarker UPDRS-3 = 35"
                )

            fig_traj.update_layout(
                title=f"📈 Vergleichende Verlaufskurve für {selected_updrs_part}",
                xaxis=dict(tickvals=months, title="Monate"),
                yaxis_title="UPDRS Score"
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        with col_t2:
            pred_df = pd.DataFrame({"Ziel": TARGET_COLS, "Score": np.round(prediction, 2)})
            fig_bar = px.bar(
                pred_df,
                x="Ziel",
                y="Score",
                color="Score",
                color_continuous_scale='Viridis',
                title="Prognoseverteilung über alle UPDRS-Dimensionen"
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# TAB 3 — XAI
# =============================================================================
with tab3:
    st.subheader("🔬 Klinische Entscheidungsfaktoren und XAI")
    st.caption(
        "SHAP-Werte zeigen, welche Merkmale die individuelle Modellprognose für UPDRS-3 nach 24 Monaten "
        "nach oben oder unten verschieben."
    )

    if st.button("🧠 Klinische Auswirkungsanalyse starten", use_container_width=True, type='primary'):
        with st.spinner("SHAP-Werte werden berechnet. Bei Full-Feature-Modellen kann dies dauern..."):
            target_idx = TARGET_COLS.index('updrs_3_plus_24')

            try:
                def predict_fn(X):
                    df = pd.DataFrame(X, columns=all_features)
                    return xgb_model.predict(df)[:, target_idx].astype(float)

                # Kleiner, deterministischer Hintergrund für stabile Laufzeiten im Dashboard.
                background_size = min(100, len(X_test_full))
                background = shap.sample(X_test_full, background_size, random_state=42)
                n_evals = 2 * len(all_features) + 1
                explainer = shap.Explainer(predict_fn, background, max_evals=n_evals)
                shap_values = explainer(patient_data)
                impacts = shap_values[0].values

                shap_df = pd.DataFrame({
                    'Feature': all_features,
                    'Label': [fmt(feature) for feature in all_features],
                    'Impact': impacts,
                    'Value': patient_data.iloc[0].values
                })
                shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                top_shap = shap_df.sort_values(by='Abs_Impact', ascending=False).head(12).copy()
                top_shap['Auswirkung'] = top_shap['Impact'].apply(
                    lambda value: 'Prognose erhöhend' if value > 0 else 'Prognose senkend/stabilisierend'
                )

                fig_shap = px.bar(
                    top_shap.sort_values(by='Impact'),
                    x="Impact",
                    y="Label",
                    orientation='h',
                    color="Auswirkung",
                    color_discrete_map={
                        'Prognose erhöhend': '#ef4444',
                        'Prognose senkend/stabilisierend': '#22c55e'
                    },
                    title="Die 12 einflussreichsten Merkmale für diesen Patienten"
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                st.divider()
                bio_cols_shap = get_top_bio_features(tuple(all_features), tuple(xgb_model.feature_importances_), top_n=50)
                col_shap1, col_shap2 = st.columns(2)

                with col_shap1:
                    st.markdown("**📊 Patienten-spezifische Biomarker-Werte (lokale Top 5)**")
                    st.caption("Biologische Merkmale mit dem größten lokalen SHAP-Einfluss im Vergleich zum Populationsdurchschnitt.")

                    local_bio_shap = shap_df[shap_df['Feature'].isin(bio_cols_shap)].copy()
                    local_top_5_bio = local_bio_shap.sort_values(by='Abs_Impact', ascending=False).head(5)
                    top_5_local_feats = local_top_5_bio['Feature'].tolist()

                    fig_local_bar = go.Figure()
                    fig_local_bar.add_trace(go.Bar(
                        name=f'Patient {selected_patient}',
                        x=[fmt(feature) for feature in top_5_local_feats],
                        y=local_top_5_bio['Value'].values,
                        marker_color='#0d6efd'
                    ))
                    fig_local_bar.add_trace(go.Bar(
                        name='Populationsdurchschnitt',
                        x=[fmt(feature) for feature in top_5_local_feats],
                        y=X_test_full[top_5_local_feats].mean().values,
                        marker_color='gray',
                        opacity=0.5
                    ))
                    fig_local_bar.update_layout(
                        barmode='group',
                        margin=dict(t=30, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_local_bar, use_container_width=True)

                with col_shap2:
                    st.markdown("**🕸️ Global wichtigste Biomarker des Modells**")
                    st.caption("Biologische Merkmale mit hoher globaler Modellbedeutung; Vergleich Patient vs. Population.")

                    imp_df = pd.DataFrame({'Feature': all_features, 'Importance': xgb_model.feature_importances_})
                    top_10_global_bio = (
                        imp_df[imp_df['Feature'].isin(bio_cols_shap)]
                        .sort_values(by='Importance', ascending=False)
                        .head(10)['Feature']
                        .tolist()
                    )

                    fig_global_radar = go.Figure()
                    fig_global_radar.add_trace(go.Scatterpolar(
                        r=X_test_full[top_10_global_bio].mean().values,
                        theta=[fmt(feature) for feature in top_10_global_bio],
                        fill='toself',
                        name='Populationsdurchschnitt',
                        line_color='gray'
                    ))
                    fig_global_radar.add_trace(go.Scatterpolar(
                        r=patient_data[top_10_global_bio].iloc[0].values,
                        theta=[fmt(feature) for feature in top_10_global_bio],
                        fill='toself',
                        name=f'Patient {selected_patient}',
                        line_color='#8b5cf6'
                    ))
                    fig_global_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
                        showlegend=True,
                        margin=dict(t=30, b=30)
                    )
                    st.plotly_chart(fig_global_radar, use_container_width=True)

                st.success("✔️ SHAP-Analyse abgeschlossen")
                st.session_state.shap_context = top_shap[['Feature', 'Impact', 'Value']].to_string(index=False)

            except Exception as exc:
                st.error(f"Berechnungsfehler: {exc}")
                st.info(
                    "Falls die Berechnung zu langsam ist, sollten SHAP-Werte im Notebook vorab berechnet "
                    "und als Dashboard-Asset gespeichert werden."
                )

# =============================================================================
# TAB 4 — CLINICAL ACTION HINTS
# =============================================================================
with tab4:
    st.subheader("💊 Klinische Handlungshinweise")
    st.caption(
        "Die folgenden Hinweise sind risikobasierte Entscheidungshilfen. Sie sind keine automatische Diagnose, "
        "keine Dosierungsanweisung und keine verbindliche Therapieempfehlung."
    )

    if st.session_state.pred_context is None:
        st.warning("Bitte generieren Sie zuerst eine Vorhersage auf der Registerkarte **Krankheitsverlauf**.")
    else:
        score = float(st.session_state.pred_context)
        risk_level = classify_updrs3_risk(score)
        st.write(f"**Prognostizierter UPDRS-3 Score nach 24 Monaten:** {score:.1f}")
        st.write(f"**Risikoklasse:** {risk_level}")

        if score >= UPDRS3_RISK_THRESHOLD:
            st.markdown("""
            <div class='alert-card alert-high'>
                <strong>🟥 Hohes modellbasiertes motorisches Risiko</strong><br>
                Eine priorisierte neurologische Verlaufskontrolle könnte in Betracht gezogen werden. 
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            **Mögliche klinische Anschlussfragen:**
            - Zeigt der Patient klinisch eine relevante motorische Verschlechterung?
            - Stimmen Modellprognose, aktuelle Untersuchung und Patientensymptomatik überein?
            - Gibt es behandelbare Faktoren wie Adhärenzprobleme, Nebenwirkungen, Stürze oder funktionelle Einschränkungen?
            - Sind engmaschigere Kontrollen oder rehabilitative Maßnahmen sinnvoll?
            """)
        elif score >= UPDRS3_MODERATE_THRESHOLD:
            st.markdown("""
            <div class='alert-card alert-mid'>
                <strong>🟨 Mittleres modellbasiertes motorisches Risiko</strong><br>
                Eine engere Verlaufsbeobachtung könnte sinnvoll sein. 
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            **Mögliche klinische Anschlussfragen:**
            - Entwickelt sich der UPDRS-3 Score konsistent in Richtung einer Verschlechterung?
            - Gibt es Hinweise auf zunehmende Einschränkungen im Alltag?
            - Könnten Bewegungstherapie, Training oder Monitoring intensiviert werden?
            """)
        else:
            st.markdown("""
            <div class='alert-card alert-low'>
                <strong>🟩 Niedriges modellbasiertes motorisches Risiko</strong><br>
                Die Prognose spricht für einen vergleichsweise stabileren Verlauf im Modellfenster. Regelmäßiges Monitoring und 
                etablierte unterstützende Maßnahmen sollten dennoch fortgeführt werden.
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            **Mögliche klinische Anschlussfragen:**
            - Bleiben motorische und nicht-motorische Symptome im Verlauf stabil?
            - Sind regelmäßige Kontrolltermine und Aktivitätsprogramme ausreichend etabliert?
            - Gibt es neue Beschwerden, die unabhängig von der Modellprognose bewertet werden müssen?
            """)

# =============================================================================
# TAB 5 — REPORT
# =============================================================================
with tab5:
    st.subheader("📄 Bericht herunterladen")

    if st.session_state.current_prediction is None:
        st.warning("Es muss zuerst eine Vorhersage generiert werden, um einen Bericht zu erstellen.")
    else:
        prediction = st.session_state.current_prediction
        updrs3_24 = float(prediction[TARGET_COLS.index('updrs_3_plus_24')])
        risk_level = classify_updrs3_risk(updrs3_24)

        report_df = pd.DataFrame({
            "Ziel": TARGET_COLS,
            "Prognostizierter Score": np.round(prediction, 2)
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        summary_text = f"""Parkinson CDSS – Forschungsbericht

Patienten-ID: {selected_patient}
Modell: XGBoost Full-Feature Modell
Biomarkerbasierter Risikotyp: {st.session_state.get('subtype_name', 'Nicht analysiert')}
UPDRS-3 Prognose nach 24 Monaten: {updrs3_24:.2f}
Risikoklasse: {risk_level}
Zeit bis zum UPDRS-3-Risikomarker: {st.session_state.get('crossing_month', 'Nicht berechnet')}

Alle Zielprognosen:
{report_df.to_string(index=False)}

SHAP-Kontext:
{st.session_state.get('shap_context', 'Noch keine SHAP-Analyse durchgeführt.')}

Hinweis:
Dieses System ist ein Forschungsprototyp. Die Prognosen und Hinweise dienen der klinischen Entscheidungsunterstützung
und ersetzen keine ärztliche Diagnose, keine klinische Beurteilung und keine Therapieentscheidung.
"""

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Prognosewerte als CSV herunterladen",
                data=csv,
                file_name=f"patient_{selected_patient}_prognose.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_r2:
            st.download_button(
                "⬇️ Zusammenfassung als TXT herunterladen",
                data=summary_text.encode('utf-8'),
                file_name=f"patient_{selected_patient}_cdss_report.txt",
                mime="text/plain",
                use_container_width=True
            )

# =============================================================================
# TAB 6 — MODEL CARD
# =============================================================================
with tab6:
    st.subheader("ℹ️ Modellinformationen und methodische Grenzen")

    st.markdown("""
    **Modelltyp:** XGBoost-Regressor im Full-Feature-Setup  
    **Eingabedaten:** klinische Verlaufsdaten, Proteomik-Features und Peptidomik-Features  
    **Zielvariablen:** UPDRS 1–4 für 0, 6, 12 und 24 Monate  
    **Validierungslogik:** patientenweise Trennung über GroupKFold im Notebook-Workflow  
    **Erklärbarkeit:** lokale und globale SHAP-basierte Modellinterpretation  
    **Dashboard-Ziel:** visuelle Entscheidungsunterstützung, Risikostratifizierung und transparente Modellinterpretation
    """)

    st.markdown("""
    **Wichtige Limitationen:**
    - Die Vorhersagen stammen aus einem retrospektiven Machine-Learning-Modell und sind nicht klinisch prospektiv validiert.
    - What-if-Simulationen sind Sensitivitätsanalysen und dürfen nicht als kausale Therapieeffekte interpretiert werden.
    - Biomarker-Cluster sind datengetriebene Risikoprofile und keine ärztlich diagnostizierten Parkinson-Subtypen.
    - Schwellenwerte im Dashboard dienen der Priorisierung und müssen klinisch kontextualisiert werden.
    - Die finale Entscheidung liegt immer beim medizinischen Fachpersonal.
    """)

    st.markdown("**Technische Übersicht:**")
    model_info = pd.DataFrame({
        "Komponente": [
            "Anzahl Testbeobachtungen",
            "Anzahl Features",
            "Anzahl Zielvariablen",
            "Risikomarker",
            "KMeans-Biomarkerfeatures"
        ],
        "Wert": [
            len(X_test_full),
            len(all_features),
            len(TARGET_COLS),
            f"UPDRS-3 ≥ {UPDRS3_RISK_THRESHOLD:.0f}",
            len(get_top_bio_features(tuple(all_features), tuple(xgb_model.feature_importances_), top_n=50))
        ]
    })

    model_info["Wert"] = model_info["Wert"].astype(str)
    st.dataframe(model_info, use_container_width=True, hide_index=True)

# =============================================================================
# CHATBOT
# =============================================================================
st.divider()
st.subheader("💬 Erklärender CDSS-Assistent")
st.caption(
    "Der Assistent erklärt Modellprognosen, SHAP-Faktoren und Risikohinweise. "
    "Er stellt keine Diagnose und gibt keine verbindlichen Therapieanweisungen."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Stellen Sie eine Frage zur Modellprognose, SHAP-Erklärung oder Risikostratifizierung ..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    shap_data = st.session_state.get('shap_context', 'Noch keine SHAP-Analyse durchgeführt.')
    pred_score = st.session_state.get('pred_context', 'Keine Vorhersage generiert.')

    sys_prompt = f"""
    Sie sind ein erklärender Assistent innerhalb eines prototypischen klinischen Entscheidungsunterstützungssystems
    für die Parkinson-Krankheit. Sie sind kein behandelnder Arzt und ersetzen keine ärztliche Beurteilung.

    Ihre Aufgabe ist es, Modellprognosen, SHAP-Erklärungen und biomarkerbasierte Risikohinweise verständlich,
    vorsichtig und akademisch zu erklären. Sie dürfen keine definitive Diagnose, keine Dosierungsanweisung und
    keine verbindliche Therapieempfehlung geben.

    [PATIENTENPROFIL UND ML-ERGEBNISSE]
    - Patienten-ID: {selected_patient}
    - Prognostizierter motorischer Score nach 24 Monaten (UPDRS-3): {pred_score}
    - Modellbasierter biomarkerbasierter Risikotyp: {st.session_state.get('subtype_name', 'Nicht analysiert')}
    - Zeit bis zum heuristischen UPDRS-3-Risikomarker: {st.session_state.get('crossing_month', 'Unbekannt')}
    - Risikoklasse: {st.session_state.get('risk_level', 'Nicht berechnet')}

    [ERKLÄRBARE KI (SHAP) - Faktoren, die die Modellprognose beeinflussen]
    {shap_data}

    [REGELN]
    1. Antworten Sie professionell, verständlich und vorsichtig.
    2. Erklären Sie, welche Modellfaktoren die Prognose erhöhen oder senken, falls SHAP-Daten vorhanden sind.
    3. Verwenden Sie Formulierungen wie "könnte", "sollte geprüft werden", "weist modellbasiert darauf hin".
    4. Geben Sie keine definitive Diagnose, keine Medikamentendosis und keine verbindliche Therapieentscheidung.
    5. Verweisen Sie bei medizinischen Entscheidungen immer auf die finale Beurteilung durch medizinisches Fachpersonal.
    6. Wenn keine Vorhersage oder keine SHAP-Analyse vorliegt, weisen Sie transparent darauf hin.
    """

    with st.chat_message("assistant"):
        with st.spinner("Modellkontext wird zusammengefasst..."):
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

                lc_messages = [SystemMessage(content=sys_prompt)]
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        lc_messages.append(HumanMessage(content=message["content"]))
                    else:
                        lc_messages.append(AIMessage(content=message["content"]))

                response = llm.invoke(lc_messages)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as exc:
                st.error(f"API-Fehler: {exc}")