"""
taste_prediction.py
=========================
(Rename of app_taste_prediction.py -> the sidebar now shows "taste prediction".)

A SEPARATE Streamlit app from app.py (your original Wine/Coffee quality app
is untouched). This file contains sequential modules on one page
(no dropdown to pick between them — they all render one after another):

  1. 🧪  5-Taste Prediction (Salt / Sweet / Umami / Bitter / Sour)
         Input methods:
           📡 Live Samples  — reads a line from the Arduino hardware (serial)
           🎯 Pick a Sample — five quick-picks or the full ChemTastesDB list
           🔬 Enter molecular measurements manually
  2. 🌿  Sample Data-Quality — transparent, formula-based reliability score
         built from ChemTastesDB's own documented record-quality fields.
  3. 🍷  Wine Sample Quality
  4. ☕  Coffee Sample Quality

Run with:
    pip install pyserial
    streamlit run taste_prediction.py
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import serial
    from serial.tools import list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASTE_MODELS_DIR = os.path.join(BASE_DIR, "models", "taste_models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
sys.path.insert(0, TASTE_MODELS_DIR)

from smiles_featurizer import featurize_dataframe, FEATURE_COLUMNS, extract_descriptors  # noqa: E402

# All charts live in chart_helpers.py (one function per chart).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_helpers as ch  # noqa: E402


def show_chart(fig):
    """Display a matplotlib figure from chart_helpers and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------
# Input-mode labels (order here = order on screen)
# ------------------------------------------------------------
MODE_LIVE = "📡 Live Samples"
MODE_SAMPLE = "🎯 Pick a Sample"
MODE_MANUAL = "🔬 Enter molecular measurements manually"


def read_arduino_line(port, baud, timeout_s=8):
    """Open the serial port, wait for the Arduino to reset, return one full line."""
    with serial.Serial(port, baud, timeout=timeout_s) as ser:
        time.sleep(2)  # Arduino auto-resets when the port opens
        ser.reset_input_buffer()
        return ser.readline().decode("utf-8", errors="ignore").strip()


def build_live_feature_vector(line, feature_stats):
    """Accepts either:
       - JSON with a "smiles" key:   {"smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"}
       - JSON with descriptor keys:  {"n_carbon": 8, "n_oxygen": 2, ...}
         (any missing descriptor falls back to the training-set average)
       - a plain SMILES string
    """
    line = line.strip()
    try:
        data = json.loads(line)
    except ValueError:
        data = None

    if isinstance(data, dict):
        if "smiles" in data:
            row = extract_descriptors(str(data["smiles"]).strip())
            return pd.DataFrame([row])[FEATURE_COLUMNS]
        row = {c: float(feature_stats.loc[c, "mean"]) for c in FEATURE_COLUMNS}
        for k, v in data.items():
            if k in FEATURE_COLUMNS:
                row[k] = float(v)
        if "n_metal_atoms" not in data:
            row["n_metal_atoms"] = (row["n_sodium"] + row["n_potassium"]
                                    + row["n_calcium"] + row["n_magnesium"])
        if "has_metal_ion" not in data:
            row["has_metal_ion"] = float(row["n_metal_atoms"] > 0)
        if "has_sulfur" not in data:
            row["has_sulfur"] = float(row["n_sulfur"] > 0)
        if "has_nitrogen" not in data:
            row["has_nitrogen"] = float(row["n_nitrogen"] > 0)
        if "has_halogen" not in data:
            row["has_halogen"] = float((row["n_fluorine"] + row["n_chlorine"]
                                        + row["n_bromine"] + row["n_iodine"]) > 0)
        if "charge_flag" not in data:
            row["charge_flag"] = float(row["n_positive_charge_symbols"] > 0
                                       or row["n_negative_charge_symbols"] > 0)
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    # not JSON -> treat the whole line as a SMILES string
    row = extract_descriptors(line)
    return pd.DataFrame([row])[FEATURE_COLUMNS]


TASTES = ["Salty", "Sweet", "Umami", "Bitter", "Sour"]
TASTE_EMOJI = {"Salty": "🧂", "Sweet": "🍬", "Umami": "🍄", "Bitter": "☕", "Sour": "🍋"}
N_TREES = 400  # matches n_estimators in train_taste_models.py

# The fixed "Sample 1..5" quick-picks used by BOTH the 5-Taste Prediction
# module and the Sample Data-Quality module, so the same five samples mean
# the same five real ChemTastesDB molecules everywhere in the app.
SAMPLE_MOLECULES = {
    "Sample 1": {"name": "Caffeine", "taste": "Bitter",
                 "smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"},
    "Sample 2": {"name": "D-Glucose", "taste": "Sweet",
                 "smiles": "OCC1OC(O)C(O)C(O)C1O"},
    "Sample 3": {"name": "Citric acid", "taste": "Sour",
                 "smiles": "OC(=O)CC(O)(CC(O)=O)C(O)=O"},
    "Sample 4": {"name": "Sodium chloride", "taste": "Salty",
                 "smiles": "[Na+].[Cl-]"},
    "Sample 5": {"name": "Adenosine 5'-monophosphate", "taste": "Umami",
                 "smiles": "Nc1ncnc2c1N=CN2C1OC(COP(O)(O)=O)C(O)C1O"},
}

st.set_page_config(
    page_title="Artificial Taste Perception — Full Suite",
    page_icon="👅",
    layout="wide",
)

from auth_history import require_login, log_prediction  # noqa: E402

require_login()

# ============================================================
# SHARED LOADERS
# ============================================================


@st.cache_resource
def load_taste_models():
    models = {}
    for taste in TASTES:
        path = os.path.join(TASTE_MODELS_DIR, f"{taste.lower()}_model.pkl")
        models[taste] = joblib.load(path)
    return models


@st.cache_data
def load_taste_dataset():
    return pd.read_csv(os.path.join(DATASETS_DIR, "ChemTastesDB_5basic_tastes_ML_ready.csv"))


@st.cache_data
def load_feature_stats():
    """Real min/max/mean per descriptor, computed from the same training
    table used by train_taste_models.py — used only to set sensible
    slider/number-input ranges and defaults below. No values are invented."""
    df = load_taste_dataset()
    clean = df[df["smiles_syntax_valid"] == True].copy()  # noqa: E712
    clean = clean[clean["stereo_label_conflict"] == False].copy()  # noqa: E712
    clean = clean.drop_duplicates(subset="canonical_smiles", keep="first")
    X = featurize_dataframe(clean, smiles_col="canonical_smiles")
    X["basic_taste_5class"] = clean["basic_taste_5class"].values
    stats = X[FEATURE_COLUMNS].describe().T
    return X, stats


@st.cache_resource
def load_wine_model():
    model = joblib.load(os.path.join(BASE_DIR, "notebooks", "random_forest_model.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "notebooks", "feature_names.pkl"))
    try:
        import shap
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None
    return model, feature_names, explainer


@st.cache_data
def load_wine_dataset():
    """WineQT, used only to show how a sample compares with real wines."""
    return pd.read_csv(os.path.join(DATASETS_DIR, "WineQT.csv"))


COFFEE_DIMS = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
               "Clean.Cup", "Sweetness"]


@st.cache_data
def load_coffee_dataset():
    """Coffee cupping data, used only to show how a sample compares with real
    coffees. Returns None if no coffee file with the expected columns exists."""
    for path in (os.path.join(BASE_DIR, "data excel", "Coffee data.csv"),
                 os.path.join(DATASETS_DIR, "coffee_quality.csv")):
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if set(COFFEE_DIMS + ["Total.Cup.Points"]) <= set(df.columns):
                return df[df["Total.Cup.Points"] > 0]
    return None


taste_models = load_taste_models()
feature_table, feature_stats = load_feature_stats()

st.markdown(
    """
    <style>
    .hero-wrap {
        position: relative;
        text-align: center;
        padding: 2.2rem 1.5rem 1.8rem 1.5rem;
        margin: 0.5rem 0 1rem 0;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(249,115,22,0.16), rgba(168,85,247,0.14) 50%, rgba(20,184,166,0.14));
        border: 1px solid rgba(249,115,22,0.45);
        box-shadow: 0 12px 40px rgba(249,115,22,0.18);
    }
    .hero-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        margin-bottom: 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #fff;
        background: linear-gradient(90deg, #F97316, #EC4899);
    }
    .hero-title {
        font-size: clamp(1.7rem, 3.6vw, 3rem);
        font-weight: 900;
        line-height: 1.2;
        margin: 0 0 0.8rem 0;
        background: linear-gradient(90deg, #F97316 0%, #EC4899 40%, #A855F7 70%, #14B8A6 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }
    .hero-accent {
        width: 120px;
        height: 5px;
        margin: 0 auto 0.9rem auto;
        border-radius: 999px;
        background: linear-gradient(90deg, #F97316, #EC4899, #14B8A6);
    }
    .hero-sub {
        font-size: 1.05rem;
        font-weight: 500;
        opacity: 0.85;
    }
    </style>
    <div class="hero-wrap">
        <div class="hero-badge">👅 E-Tongue · Explainable AI · Machine Learning</div>
        <div class="hero-title">🧠 Computational Modelling of Artificial Taste Perception using Machine Learning with Explainable AI, Inspired by the E-Tongue</div>
        <div class="hero-accent"></div>
        <div class="hero-sub">🧪 5-Taste Prediction &nbsp;·&nbsp; 🍷 Wine Quality &nbsp;·&nbsp; ☕ Coffee Quality &nbsp;·&nbsp; 🌿 Sample Data-Quality</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "🧪 5-taste prediction · 🍷 Wine sample quality · ☕ Coffee sample quality · "
    "🌿 sample data-quality — all on one page, no mode selector."
)
st.markdown("---")

# ============================================================
# MODULE 1 — 5-TASTE PREDICTION
# ============================================================

st.header("👅 5-Taste Prediction")
st.write(
    "Enter the molecular measurements below the same way the Wine module further "
    "down asks for *fixed acidity*, *chlorides*, *sulphates*, etc. Each field here "
    "is a real molecular descriptor extracted from ChemTastesDB SMILES structures "
    "and is exactly what the 5 taste models were trained on — nothing here is a "
    "placeholder."
)

input_mode = st.radio(
    "Input method",
    [MODE_LIVE, MODE_SAMPLE, MODE_MANUAL],
    horizontal=True,
)

# ------------------------------------------------------------
# Build default/min/max from REAL dataset statistics
# ------------------------------------------------------------


def stat(col, which):
    return float(feature_stats.loc[col, which])


def num_input(col, label, step, fmt=None, col_widget=st, help_text=None):
    kwargs = dict(
        min_value=float(0.0),
        max_value=float(max(stat(col, "max"), 1.0)),
        value=round(float(stat(col, "mean")), 3),
        step=step,
        help=help_text,
    )
    if fmt:
        kwargs["format"] = fmt
    return col_widget.number_input(label, **kwargs)


manual_values = {}

if input_mode == MODE_MANUAL:

    st.subheader("🧬 Primary molecular measurements")
    st.caption(
        "These 12 descriptors carry the most predictive weight across the 5 taste "
        "models (ranked by trained RandomForest feature importance). Defaults are "
        "set to the ChemTastesDB training-set average for each field."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        manual_values["n_carbon"] = num_input("n_carbon", "Carbon Atom Count", 1.0, col_widget=c1)
        manual_values["o_to_c_ratio"] = num_input("o_to_c_ratio", "Oxygen : Carbon Ratio", 0.01, "%.3f", c1)
        manual_values["approx_ring_count"] = num_input("approx_ring_count", "Ring Count", 1.0, col_widget=c1)
        manual_values["n_to_c_ratio"] = num_input("n_to_c_ratio", "Nitrogen : Carbon Ratio", 0.01, "%.3f", c1)
    with c2:
        manual_values["n_heavy_atoms"] = num_input("n_heavy_atoms", "Total Heavy Atoms", 1.0, col_widget=c2)
        manual_values["smiles_length"] = num_input("smiles_length", "Structure Complexity (SMILES length)", 1.0, col_widget=c2)
        manual_values["approx_mol_weight_heavy_atoms"] = num_input(
            "approx_mol_weight_heavy_atoms", "Molecular Weight (heavy atoms)", 1.0, "%.2f", c2)
        manual_values["n_nitrogen"] = num_input("n_nitrogen", "Nitrogen Atom Count", 1.0, col_widget=c2)
    with c3:
        manual_values["n_double_bonds"] = num_input("n_double_bonds", "Double Bond Count", 1.0, col_widget=c3)
        manual_values["n_oxygen"] = num_input("n_oxygen", "Oxygen Atom Count", 1.0, col_widget=c3)
        manual_values["n_branches"] = num_input("n_branches", "Branching (side chains)", 1.0, col_widget=c3)
        manual_values["n_aromatic_symbols"] = num_input("n_aromatic_symbols", "Aromatic Atom Count", 1.0, col_widget=c3)

    with st.expander("⚙️ Advanced / ionic-content measurements (sodium, chloride & more)"):
        st.caption(
            "Saltiness in ChemTastesDB is driven almost entirely by ionic species "
            "(Na⁺, K⁺, Cl⁻), not carbon-skeleton chemistry — these fields matter "
            "specifically for the Salt model."
        )
        a1, a2, a3 = st.columns(3)
        with a1:
            manual_values["n_sodium"] = num_input("n_sodium", "Sodium Atom Count", 1.0, col_widget=a1)
            manual_values["n_potassium"] = num_input("n_potassium", "Potassium Atom Count", 1.0, col_widget=a1)
            manual_values["n_chlorine"] = num_input("n_chlorine", "Chlorine Atom Count", 1.0, col_widget=a1)
        with a2:
            manual_values["n_calcium"] = num_input("n_calcium", "Calcium Atom Count", 1.0, col_widget=a2)
            manual_values["n_magnesium"] = num_input("n_magnesium", "Magnesium Atom Count", 1.0, col_widget=a2)
            manual_values["n_sulfur"] = num_input("n_sulfur", "Sulfur Atom Count", 1.0, col_widget=a2)
        with a3:
            manual_values["n_fluorine"] = num_input("n_fluorine", "Fluorine Atom Count", 1.0, col_widget=a3)
            manual_values["n_bromine"] = num_input("n_bromine", "Bromine Atom Count", 1.0, col_widget=a3)
            manual_values["n_iodine"] = num_input("n_iodine", "Iodine Atom Count", 1.0, col_widget=a3)

        b1, b2, b3 = st.columns(3)
        with b1:
            manual_values["n_phosphorus"] = num_input("n_phosphorus", "Phosphorus Atom Count", 1.0, col_widget=b1)
        with b2:
            manual_values["n_triple_bonds"] = num_input("n_triple_bonds", "Triple Bond Count", 1.0, col_widget=b2)
        with b3:
            manual_values["s_to_c_ratio"] = num_input("s_to_c_ratio", "Sulfur : Carbon Ratio", 0.01, "%.3f", b3)

        c1b, c2b = st.columns(2)
        with c1b:
            manual_values["n_positive_charge_symbols"] = num_input(
                "n_positive_charge_symbols", "Positive Charge Sites", 1.0, col_widget=c1b)
            manual_values["n_bracket_atoms"] = num_input("n_bracket_atoms", "Bracket-Notation Atoms", 1.0, col_widget=c1b)
        with c2b:
            manual_values["n_negative_charge_symbols"] = num_input(
                "n_negative_charge_symbols", "Negative Charge Sites", 1.0, col_widget=c2b)

    # ---- derive the flag/composite columns exactly the way
    # ---- smiles_featurizer.extract_descriptors() derives them ----
    manual_values["n_metal_atoms"] = (
        manual_values["n_sodium"] + manual_values["n_potassium"]
        + manual_values["n_calcium"] + manual_values["n_magnesium"]
    )
    manual_values["has_metal_ion"] = float(manual_values["n_metal_atoms"] > 0)
    manual_values["has_sulfur"] = float(manual_values["n_sulfur"] > 0)
    manual_values["has_nitrogen"] = float(manual_values["n_nitrogen"] > 0)
    manual_values["has_halogen"] = float(
        (manual_values["n_fluorine"] + manual_values["n_chlorine"]
         + manual_values["n_bromine"] + manual_values["n_iodine"]) > 0
    )
    manual_values["charge_flag"] = float(
        (manual_values["n_positive_charge_symbols"] > 0)
        or (manual_values["n_negative_charge_symbols"] > 0)
    )

    st.caption(
        "🔗 `has_metal_ion`, `has_sulfur`, `has_nitrogen`, `has_halogen`, `charge_flag`, "
        "and `n_metal_atoms` are **derived automatically** from the counts above, using "
        "the exact same rule `smiles_featurizer.py` uses when parsing a SMILES string — "
        "they are not separately editable so the feature vector always stays internally "
        "consistent."
    )

    feature_vector = pd.DataFrame([{col: manual_values.get(col, 0.0) for col in FEATURE_COLUMNS}])
    smiles_input_for_display = None

elif input_mode == MODE_SAMPLE:
    # Same "select a section" pattern app.py uses to switch between
    # 🍷 Wine / ☕ Coffee — here it switches between the 5 quick samples
    # and the full ChemTastesDB molecule list.
    section_col, _ = st.columns([1, 1])
    with section_col:
        sample_section = st.selectbox(
            "Select section",
            ["🎯 Pick Sample (1–5)", "📋 Pick from Sample Data-Quality List"],
            key="taste_sample_section",
        )
    taste_dataset_for_picker = load_taste_dataset()

    if sample_section == "🎯 Pick Sample (1–5)":
        sample_col, _ = st.columns([1, 1])
        with sample_col:
            sample_key = st.selectbox("Sample", list(SAMPLE_MOLECULES.keys()), key="taste_sample_15")
        picked = SAMPLE_MOLECULES[sample_key]
        # (the "Sample 1 = Caffeine (labeled taste: Bitter)" caption was removed)
        smiles_input_for_display = picked["smiles"]
    else:
        all_names = sorted(taste_dataset_for_picker["name"].dropna().unique().tolist())
        default_index = all_names.index("Caffeine") if "Caffeine" in all_names else 0
        picked_name = st.selectbox("Molecule (full ChemTastesDB list)", all_names,
                                   index=default_index, key="taste_full_list")
        picked_row = taste_dataset_for_picker[taste_dataset_for_picker["name"] == picked_name].iloc[0]
        smiles_input_for_display = picked_row["canonical_smiles"]

    if smiles_input_for_display and str(smiles_input_for_display).strip():
        row = extract_descriptors(str(smiles_input_for_display).strip())
        feature_vector = pd.DataFrame([row])[FEATURE_COLUMNS]
    else:
        feature_vector = None

else:  # ---------------- MODE_LIVE: Arduino hardware ----------------
    smiles_input_for_display = None
    feature_vector = None

    st.subheader("📡 Live Sample — read from Arduino")
    if not SERIAL_OK:
        st.error("The `pyserial` package is missing. Run `pip install pyserial` and restart the app.")
    else:
        ports = [p.device for p in list_ports.comports()]
        lc1, lc2 = st.columns(2)
        with lc1:
            port = st.selectbox("Arduino COM port", ports if ports else ["(no port found)"], key="live_port")
        with lc2:
            baud = st.selectbox("Baud rate", [9600, 19200, 38400, 57600, 115200], index=0, key="live_baud")

        if st.button("📡 Read from Arduino", key="live_read_btn"):
            if not ports:
                st.error("No Arduino detected. Plug it in, close the Arduino Serial Monitor, and reload.")
            else:
                try:
                    with st.spinner("Reading from Arduino..."):
                        st.session_state["live_line"] = read_arduino_line(port, baud)
                except Exception as e:
                    st.error(f"Could not read from {port}: {e}")

    live_line = st.session_state.get("live_line", "")
    if live_line:
        st.caption("Latest line received from the Arduino:")
        st.code(live_line)
        try:
            feature_vector = build_live_feature_vector(live_line, feature_stats)
            st.success("Live reading converted. Click **Predict Taste Profile** below.")
        except Exception as e:
            st.error(f"Could not understand the Arduino data: {e}")
    else:
        st.info("Click **Read from Arduino** to capture a live sample.")

st.markdown("---")

if st.button("🧠 Predict Taste Profile", use_container_width=True, type="primary"):
    if feature_vector is None:
        st.error("Please provide a sample first (read from the Arduino, pick a sample, or enter measurements).")
    else:
        feature_vector = feature_vector[FEATURE_COLUMNS]

        scores = {}
        for taste, model in taste_models.items():
            proba = model.predict_proba(feature_vector)[0][1]
            scores[taste] = float(proba)

        dominant_taste = max(scores, key=scores.get)
        log_prediction("5-Taste Prediction", f"Dominant taste: {dominant_taste} ({scores[dominant_taste]*100:.1f}%)")

        st.markdown("---")
        st.subheader("🎯 Predicted Taste Confidence")

        score_df = pd.DataFrame({
            "Taste": [f"{TASTE_EMOJI[t]} {t}" for t in scores],
            "raw_taste": list(scores.keys()),
            "Confidence Score (%)": [round(v * 100, 1) for v in scores.values()],
        }).sort_values("Confidence Score (%)", ascending=False)

        chart_col, donut_col = st.columns(2)
        with chart_col:
            show_chart(ch.taste_ranked_bars(scores, dominant_taste))
        with donut_col:
            show_chart(ch.taste_donut(scores, dominant_taste))

        st.dataframe(
            score_df[["Taste", "Confidence Score (%)"]],
            use_container_width=True,
            hide_index=True,
        )

        st.success(f"**Dominant predicted taste: {TASTE_EMOJI[dominant_taste]} {dominant_taste}**")

        for t in TASTES:
            st.progress(min(int(scores[t] * 100), 100), text=f"{TASTE_EMOJI[t]} {t}: {scores[t]*100:.1f}%")

        # --------------------------------------------------
        # FORMULA + WHY (no "probability" language)
        # --------------------------------------------------
        st.markdown("---")
        st.header(f"🔍 Why '{dominant_taste}'? — Formula & Explanation")

        st.markdown(
            f"""
**How the Confidence Score is calculated**

Each taste model is a **RandomForestClassifier** made of **{N_TREES} decision
trees**, each trained on a different random resample of ChemTastesDB. Every
tree independently votes *"this taste"* or *"not this taste"* for the molecule
you entered.

$$
\\text{{Confidence Score}}(\\%) = \\frac{{\\text{{number of trees voting FOR the taste}}}}{{{N_TREES}\\ \\text{{total trees}}}} \\times 100
$$

For **{dominant_taste}**, that works out to **{scores[dominant_taste]*100:.1f}%**
— i.e. roughly **{int(round(scores[dominant_taste]*N_TREES))} out of {N_TREES}**
trees voted for {dominant_taste} given the measurements you entered.
"""
        )

        # local explanation: SHAP if available, else transparent fallback
        model = taste_models[dominant_taste]
        explanation_ready = False
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feature_vector)
            if isinstance(shap_values, list):
                contributions = shap_values[1][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                contributions = shap_values[0, :, 1]
            else:
                contributions = np.ravel(shap_values)[: len(FEATURE_COLUMNS)]
            explain_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Your Value": feature_vector.iloc[0].values,
                "Contribution": contributions,
            })
            explanation_method = "SHAP (Shapley additive contribution per feature)"
            explanation_ready = True
        except Exception:
            # Deterministic, honest fallback: global feature importance
            # weighted by how far the entered value sits from the
            # training-set mean for that feature (direction only).
            means = feature_stats["mean"]
            importances = model.feature_importances_
            deviations = (feature_vector.iloc[0].values - means[FEATURE_COLUMNS].values)
            contributions = importances * np.sign(deviations) * np.log1p(np.abs(deviations))
            explain_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Your Value": feature_vector.iloc[0].values,
                "Contribution": contributions,
            })
            explanation_method = (
                "Approximate (SHAP not installed in this environment) — "
                "global feature importance × direction/size of deviation "
                "from the ChemTastesDB training-set average"
            )
            explanation_ready = True

        if explanation_ready:
            explain_df["Absolute Impact"] = explain_df["Contribution"].abs()
            explain_df = explain_df.sort_values("Absolute Impact", ascending=False)
            top_explain = explain_df.head(8).copy()
            top_explain["Direction"] = np.where(
                top_explain["Contribution"] > 0, "⬆️ Pushes toward " + dominant_taste, "⬇️ Pushes away"
            )

            st.caption(f"Explanation method: {explanation_method}")

            ecol1, ecol2 = st.columns([1, 1])
            with ecol1:
                st.dataframe(
                    top_explain[["Feature", "Your Value", "Contribution", "Direction"]],
                    use_container_width=True,
                    hide_index=True,
                )
            with ecol2:
                show_chart(ch.contribution_waterfall(top_explain, dominant_taste))

            strongest = explain_df.iloc[0]
            direction_text = "increased" if strongest["Contribution"] > 0 else "decreased"
            st.info(
                f"💡 The strongest single factor was **{strongest['Feature']}** "
                f"(your value: {strongest['Your Value']:.3f}), which **{direction_text}** "
                f"the model's confidence in **{dominant_taste}**. This is *not* a probability "
                f"estimate in the statistical sense — it is the literal fraction of trained "
                f"decision trees inside the Random Forest that agreed on this taste."
            )

        # --------------------------------------------------
        # Dashboard: where does this molecule sit vs. the training data?
        # --------------------------------------------------
        st.markdown("---")
        st.subheader(f"📈 How does this sample compare to real {dominant_taste} molecules?")

        sample_mw = float(feature_vector.iloc[0]["approx_mol_weight_heavy_atoms"])
        sample_oc = float(feature_vector.iloc[0]["o_to_c_ratio"])
        dash_col1, dash_col2 = st.columns(2)
        with dash_col1:
            ref = feature_table[feature_table["basic_taste_5class"] == dominant_taste][
                "approx_mol_weight_heavy_atoms"]
            show_chart(ch.percentile_line(
                ref, sample_mw, dominant_taste, ch.TASTE_COLORS[dominant_taste],
                "Molecular weight (heavy atoms)",
                "Molecular weight vs. real molecules of this taste"))
        with dash_col2:
            show_chart(ch.taste_scatter(feature_table, dominant_taste, sample_mw, sample_oc))

        if scores.get("Salty", 0) > 0 or scores.get("Sour", 0) > 0:
            st.caption(
                "⚠️ Salt (16 training molecules) and Sour (49 training molecules) models "
                "were trained on very few positive examples — treat their scores as a "
                "rough directional signal, not a confident readout. See "
                "`results/taste_model_comparison_report.md` for full metrics."
            )

st.markdown("---")

# ============================================================
# MODULE — SAMPLE DATA-QUALITY
# ============================================================

st.header("🧫 Sample Data-Quality")
st.write(
    "WineQT and the CQI coffee dataset each carry an expert-assigned *sensory* "
    "quality score (0–10 / 0–100) — ChemTastesDB does not, since it records "
    "molecular taste identity, not a tasting panel's quality rating. What it "
    "**does** record, per molecule, is how well-documented that taste label is: "
    "whether the PubChem CID and CAS number are valid, whether the name is "
    "curated rather than auto-generated, whether the SMILES had a stereo-label "
    "conflict, and how many literature references back the label. That's what "
    "this module scores — a **data-quality / reliability index**, not a taste "
    "intensity rating."
)

dq_taste_df = load_taste_dataset()

dq_mode = st.radio(
    "Input method",
    ["📋 Pick a real molecule from ChemTastesDB", "✏️ Enter the reliability fields manually"],
    horizontal=True,
    key="dq_mode",
)
dq_your_taste = None  # set below when a real molecule is picked

if dq_mode == "📋 Pick a real molecule from ChemTastesDB":
    # Same "select a section" pattern as the 5-Taste Prediction module and
    # app.py's Wine/Coffee switch — Section 1 is the 5 fixed quick-picks,
    # Section 2 is the full ChemTastesDB list, unchanged from before.
    section_col_dq, _ = st.columns([1, 1])
    with section_col_dq:
        dq_section = st.selectbox(
            "Select section",
            ["🎯 Pick Sample (1–5)", "📋 Full ChemTastesDB List"],
            key="dq_sample_section",
        )

    if dq_section == "🎯 Pick Sample (1–5)":
        sample_col_dq, _ = st.columns([1, 1])
        with sample_col_dq:
            sample_key = st.selectbox("Sample", list(SAMPLE_MOLECULES.keys()), key="dq_sample_15")
        molecule_name = SAMPLE_MOLECULES[sample_key]["name"]
    else:
        all_names_dq = sorted(dq_taste_df["name"].dropna().unique().tolist())
        default_index_dq = all_names_dq.index("Caffeine") if "Caffeine" in dq_taste_df["name"].values else 0
        molecule_name = st.selectbox("Molecule", all_names_dq, index=default_index_dq, key="dq_full_list")

    row = dq_taste_df[dq_taste_df["name"] == molecule_name].iloc[0]
    valid_cid = bool(row["pubchem_cid_format_valid"]) if pd.notna(row["pubchem_cid_format_valid"]) else False
    valid_cas = bool(row["cas_number_format_valid"]) if pd.notna(row["cas_number_format_valid"]) else False
    curated_name = not bool(row["name_is_autogenerated_iupac"])
    no_stereo_conflict = not bool(row["stereo_label_conflict"])
    ref_count = 0 if pd.isna(row["reference"]) else len(str(row["reference"]).split(";"))
    dq_your_taste = row["basic_taste_5class"]
else:
    dqc1, dqc2 = st.columns(2)
    with dqc1:
        valid_cid = st.checkbox("Has a valid PubChem CID", value=True)
        valid_cas = st.checkbox("Has a valid CAS registry number", value=True)
        curated_name = st.checkbox("Name is human-curated (not auto-generated IUPAC)", value=True)
    with dqc2:
        no_stereo_conflict = st.checkbox("No stereo-label conflict", value=True)
        ref_count = st.number_input("Number of literature references citing this label", 0, 20, 3, 1)

st.markdown("---")

if st.button("🧫 Compute Sample Data-Quality", use_container_width=True, key="dq_btn"):
    weights = {
        "Valid PubChem CID": (valid_cid, 20),
        "Valid CAS number": (valid_cas, 20),
        "Curated (non-autogenerated) name": (curated_name, 20),
        "No stereo-label conflict": (no_stereo_conflict, 20),
        "Has ≥1 literature reference": (ref_count >= 1, 20),
    }
    quality_score = sum(pts for ok, pts in weights.values() if ok)
    log_prediction("Sample Data-Quality", f"Score: {quality_score} / 100")

    qcol1, qcol2 = st.columns([1, 2])
    with qcol1:
        st.metric("Sample Data-Quality", f"{quality_score} / 100")
        if quality_score >= 80:
            st.success("High reliability — well-documented label.")
        elif quality_score >= 50:
            st.warning("Moderate reliability — some documentation gaps.")
        else:
            st.error("Low reliability — treat this molecule's taste label cautiously.")
    with qcol2:
        show_chart(ch.quality_gauge(quality_score))

    # score for every record, used by the dataset-level charts further down
    all_scores = (
        dq_taste_df["pubchem_cid_format_valid"].fillna(False).astype(bool).astype(int) * 20
        + dq_taste_df["cas_number_format_valid"].fillna(False).astype(bool).astype(int) * 20
        + (~dq_taste_df["name_is_autogenerated_iupac"].astype(bool)).astype(int) * 20
        + (~dq_taste_df["stereo_label_conflict"].astype(bool)).astype(int) * 20
        + (dq_taste_df["reference"].notna()).astype(int) * 20
    )

    chk_col, tier_col = st.columns(2)
    with chk_col:
        show_chart(ch.checklist_tiles(weights))
    with tier_col:
        show_chart(ch.tier_donut(all_scores, quality_score))

    st.markdown(
        f"""
**Formula**

$$
\\text{{Data-Quality Score}} = 20 \\times \\big[ \\text{{valid CID}} \\big] + 20 \\times \\big[ \\text{{valid CAS}} \\big]
+ 20 \\times \\big[ \\text{{curated name}} \\big] + 20 \\times \\big[ \\text{{no stereo conflict}} \\big]
+ 20 \\times \\big[ \\text{{≥1 reference}} \\big]
$$

Each bracket is 1 if the check passes and 0 otherwise — this is a transparent,
rule-based index built directly from ChemTastesDB's own documented fields, not
a trained model, so there's nothing to overfit or misrepresent as a sensory rating.
"""
    )

    st.subheader("📊 Where this sits across the whole dataset")
    stack_col, note_col = st.columns(2)
    with stack_col:
        show_chart(ch.tier_stacked_by_taste(
            pd.DataFrame({"taste": dq_taste_df["basic_taste_5class"].values,
                          "score": all_scores.values}),
            dq_your_taste))
    with note_col:
        st.markdown(
            f"""
**How to read the tiers**

- ✓ **High** - score 80 or more
- ! **Moderate** - score 50 to 79
- ✗ **Low** - score below 50

The bars show what share of ChemTastesDB's {len(dq_taste_df):,} basic-taste
molecules fall in each tier, split by taste class. A ▶ marks the class of the
molecule you picked.
"""
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "5-Taste: RandomForestClassifier × 5, trained on ChemTastesDB v2.1 (CC BY-NC-ND). "
    "Wine/Coffee: models from `app.py` / `train_models.py`. Data-Quality: rule-based index "
    "over ChemTastesDB's own documented reliability fields. "
    "See `results/taste_model_comparison_report.md` and `README_taste_models.md` for "
    "full methodology and accuracy numbers."
)


# ============================================================
# MODULE — WINE SAMPLE QUALITY
# ============================================================

st.header("🍷 Wine Sample Quality")
st.write(
    "Enter the chemical sensor measurements below. This uses the same trained "
    "model as `app.py`'s wine section — only the labelling here is generalised "
    "from *quality* to **sample quality** so it reads consistently with the "
    "other modules on this page."
)

wine_model, wine_feature_names, wine_explainer = load_wine_model()
wine_dataset = load_wine_dataset()

wc1, wc2, wc3 = st.columns(3)
with wc1:
    fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4, 0.1, key="w_fa")
    volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.70, 0.01, key="w_va")
    citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.00, 0.01, key="w_ca")
    residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9, 0.1, key="w_rs")
with wc2:
    chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076, 0.001, format="%.3f", key="w_ch")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0, 1.0, key="w_fs")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 350.0, 34.0, 1.0, key="w_ts")
    density = st.number_input("Density", 0.98, 1.02, 0.9978, 0.0001, format="%.4f", key="w_de")
with wc3:
    ph = st.number_input("pH", 2.0, 5.0, 3.51, 0.01, key="w_ph")
    sulphates = st.number_input("Sulphates", 0.0, 3.0, 0.56, 0.01, key="w_su")
    alcohol = st.number_input("Alcohol (%)", 5.0, 20.0, 9.4, 0.1, key="w_al")

wine_input = pd.DataFrame([{
    "fixed acidity": fixed_acidity, "volatile acidity": volatile_acidity,
    "citric acid": citric_acid, "residual sugar": residual_sugar,
    "chlorides": chlorides, "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide, "density": density,
    "pH": ph, "sulphates": sulphates, "alcohol": alcohol,
}])[wine_feature_names]

st.markdown("---")

if st.button("🍇 Analyze Wine Sample Quality", use_container_width=True, key="wine_btn"):
    wine_prediction = int(wine_model.predict(wine_input)[0])
    log_prediction("Wine Sample Quality", f"Predicted quality: {wine_prediction} / 10")

    mcol1, mcol2 = st.columns([1, 2])
    with mcol1:
        st.metric("Predicted Sample Quality", f"{wine_prediction} / 10")
    with mcol2:
        show_chart(ch.wine_bullet(wine_prediction, wine_dataset["quality"]))

    if hasattr(wine_model, "predict_proba"):
        probabilities = wine_model.predict_proba(wine_input)[0]
        classes = wine_model.classes_
        probability_df = pd.DataFrame({"Quality": classes, "Confidence (%)": probabilities * 100})
        probability_df = probability_df.sort_values("Confidence (%)", ascending=False)

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            st.subheader("📊 Model Confidence Distribution")
            show_chart(ch.confidence_curve(
                probability_df["Quality"].to_numpy(),
                probability_df["Confidence (%)"].to_numpy(),
                wine_prediction))
        with chart_c2:
            st.subheader("🍇 Sensor Values vs. WineQT Average")
            show_chart(ch.zscore_bars(
                {n: float(wine_input.iloc[0][n]) for n in wine_feature_names},
                wine_dataset, ch.VIOLET,
                "How unusual are your sensor readings?",
                "Standard deviations from the WineQT average (0 = typical)"))

    if wine_explainer is not None:
        st.markdown("---")
        st.header("🔍 Why did the model make this prediction?")

        shap_values = wine_explainer.shap_values(wine_input)
        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                class_index = list(wine_model.classes_).index(wine_prediction)
                contributions = shap_values[0, :, class_index]
            elif shap_values.ndim == 2:
                contributions = shap_values[0]
            else:
                contributions = np.ravel(shap_values)
        else:
            contributions = np.asarray(shap_values)

        explanation_df = pd.DataFrame({
            "Feature": wine_feature_names,
            "Input Value": wine_input.iloc[0].values,
            "SHAP Contribution": contributions,
        })
        explanation_df["Absolute Impact"] = explanation_df["SHAP Contribution"].abs()
        explanation_df = explanation_df.sort_values("Absolute Impact", ascending=False)
        top_features = explanation_df.head(8).copy()

        ec1, ec2 = st.columns([1, 1])
        with ec1:
            display_df = top_features[["Feature", "Input Value", "SHAP Contribution"]].copy()
            display_df["Direction"] = np.where(display_df["SHAP Contribution"] > 0, "⬆️ Positive", "⬇️ Negative")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        with ec2:
            show_chart(ch.contribution_lollipop(top_features, f"quality {wine_prediction}"))

        strongest = explanation_df.iloc[0]
        direction_text = "increased" if strongest["SHAP Contribution"] > 0 else "decreased"
        st.info(
            f"💡 The strongest contribution came from **{strongest['Feature']}** "
            f"(input value: {strongest['Input Value']:.4f}), which **{direction_text}** "
            f"the model's confidence in sample quality {wine_prediction}."
        )

st.markdown("---")

# ============================================================
# MODULE — COFFEE SAMPLE QUALITY
# ============================================================

st.header("☕ Coffee Sample Quality")
st.write("A sensory-profile module comparing multiple cupping dimensions, generalised the same way as the wine module above.")

cc1, cc2 = st.columns(2)
with cc1:
    aroma = st.slider("Aroma", 6.0, 10.0, 7.8, 0.1, key="c_aroma")
    flavor = st.slider("Flavor", 6.0, 10.0, 7.8, 0.1, key="c_flavor")
    aftertaste = st.slider("Aftertaste", 6.0, 10.0, 7.5, 0.1, key="c_after")
    acidity = st.slider("Acidity", 6.0, 10.0, 7.6, 0.1, key="c_acid")
with cc2:
    body = st.slider("Body", 6.0, 10.0, 7.5, 0.1, key="c_body")
    balance = st.slider("Balance", 6.0, 10.0, 7.6, 0.1, key="c_bal")
    clean_cup = st.slider("Clean Cup", 6.0, 10.0, 10.0, 0.1, key="c_clean")
    sweetness = st.slider("Sweetness", 6.0, 10.0, 10.0, 0.1, key="c_sweet")

total_score = aroma + flavor + aftertaste + acidity + body + balance + clean_cup + sweetness + 15.0

st.markdown("---")

qm1, qm2 = st.columns([1, 2])
with qm1:
    st.metric("Coffee Sample Quality", f"{total_score:.2f} / 100")
    st.caption("80.0+ is the industry 'specialty coffee' cutoff (SCA cupping protocol).")
sensory_data = pd.DataFrame({
    "Sensory Dimension": ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Clean Cup", "Sweetness"],
    "Score": [aroma, flavor, aftertaste, acidity, body, balance, clean_cup, sweetness],
})
coffee_dataset = load_coffee_dataset()

with qm2:
    if coffee_dataset is not None:
        show_chart(ch.density_with_marker(
            coffee_dataset["Total.Cup.Points"], total_score, 80.0, ch.ORANGE,
            "Total cup points", "Your total vs. the coffee dataset",
            "Specialty cutoff (80)"))
    else:
        st.info("No coffee dataset file was found, so the comparison chart is skipped.")

st.subheader("👃 Sensory Component Breakdown")
bcol, tcol = st.columns(2)
with bcol:
    if coffee_dataset is not None:
        show_chart(ch.coffee_dumbbell(
            sensory_data["Sensory Dimension"].tolist(),
            sensory_data["Score"].tolist(),
            [float(coffee_dataset[d].mean()) for d in COFFEE_DIMS]))
with tcol:
    show_chart(ch.coffee_stacked_total(
        sensory_data["Sensory Dimension"].tolist(),
        sensory_data["Score"].tolist(), 15.0))

st.markdown("---")
