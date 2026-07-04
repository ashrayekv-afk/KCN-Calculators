#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud-safe dual KCN calculator app.

This version intentionally does NOT require SHAP because SHAP/numba often fails
on Streamlit Cloud Python builds. The explanation panel uses a deployment-safe
"score impact" method: it compares the current score with the score after setting
one feature at a time to the training median.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="KCN Progression-Like Calculators", layout="centered")

BASELINE_BUNDLE_PATH = Path(__file__).with_name("kcn_baseline_calculator_enhanced_model_bundle.joblib")
TWO_SCAN_BUNDLE_PATH = Path(__file__).with_name("kcn_two_scan_calculator_enhanced_model_bundle.joblib")

baseline_bundle = joblib.load(BASELINE_BUNDLE_PATH)
two_scan_bundle = joblib.load(TWO_SCAN_BUNDLE_PATH)


def predict_model_object(model_object, X):
    """Predict probability from either a single sklearn model or an ensemble dict."""
    if isinstance(model_object, dict) and model_object.get("type") == "ensemble":
        probs = []
        for kind, model in model_object["models"]:
            probs.append(model.predict_proba(X)[:, 1])
        return np.mean(np.vstack(probs), axis=0)
    return model_object.predict_proba(X)[:, 1]


def main_predict(bundle, feature_row):
    if "model_object" in bundle:
        return float(predict_model_object(bundle["model_object"], feature_row)[0])
    if "model" in bundle:
        return float(bundle["model"].predict_proba(feature_row)[0, 1])
    raise ValueError("Bundle has no supported model object.")


def risk_category(bundle, probability):
    q = bundle.get("score_quantiles", {})
    low_cut = q.get("q25", 0.25)
    high_cut = q.get("q75", 0.75)
    if probability < low_cut:
        return "Low", "More control-like than most eyes in the training cohort."
    if probability < high_cut:
        return "Intermediate", "Borderline/intermediate score relative to the training cohort."
    return "High", "More similar to the CXL/progression-like group in the training cohort."


def show_range_warnings(bundle, feature_row):
    ranges = bundle.get("training_feature_ranges", {})
    warnings = []
    for feature in bundle["features"]:
        if feature not in ranges:
            continue
        value = float(feature_row.iloc[0][feature])
        r = ranges[feature]
        display = bundle["feature_display_names"].get(feature, feature)
        if value < r["min"] or value > r["max"]:
            warnings.append(
                f"**{display}** = {value:.3g} is outside the full training range "
                f"({r['min']:.3g}–{r['max']:.3g})."
            )
        elif value < r["p01"] or value > r["p99"]:
            warnings.append(
                f"**{display}** = {value:.3g} is outside the 1st–99th percentile training range "
                f"({r['p01']:.3g}–{r['p99']:.3g})."
            )
    if warnings:
        st.warning("Some inputs are outside the training range. Interpret cautiously:\n\n" + "\n\n".join(warnings))


def score_impact_explanation(bundle, feature_row):
    """Deployment-safe explanation: compare current score to score with each feature reset to training median."""
    current = main_predict(bundle, feature_row)
    rows = []
    ranges = bundle.get("training_feature_ranges", {})
    for feature in bundle["features"]:
        if feature not in ranges:
            continue
        baseline = feature_row.copy()
        median_value = ranges[feature].get("median", float(feature_row.iloc[0][feature]))
        baseline.loc[:, feature] = median_value
        prob_at_median = main_predict(bundle, baseline)
        impact = current - prob_at_median
        rows.append({
            "Feature": bundle["feature_display_names"].get(feature, feature),
            "Current value": float(feature_row.iloc[0][feature]),
            "Training median": float(median_value),
            "Score impact": float(impact),
            "Abs impact": float(abs(impact)),
        })
    return pd.DataFrame(rows).sort_values("Abs impact", ascending=False)


def show_explanation_panel(bundle, feature_row):
    with st.expander("Why did the calculator give this score?", expanded=True):
        st.markdown(
            "This deployment-safe explanation changes one input at a time back to the training median and recalculates the score. "
            "Positive impact means the current value is pushing the score higher; negative impact means it is pushing the score lower."
        )
        ex = score_impact_explanation(bundle, feature_row).head(8)
        st.dataframe(ex[["Feature", "Current value", "Training median", "Score impact"]], width="stretch")
        st.bar_chart(ex.set_index("Feature")[["Score impact"]], width="stretch")
        st.caption("This is an explanation of model behavior, not a causal claim.")


def show_horizon_panel(bundle, feature_row):
    with st.expander("Prototype 1-, 2-, and 3-year estimates", expanded=False):
        st.markdown(
            "These are **prototype** time-horizon estimates using available follow-up interval as an event-time proxy. "
            "They are not formal survival-model probabilities."
        )
        rows = []
        for horizon, hbundle in bundle.get("horizon_models", {}).items():
            h_features = hbundle["features"]
            h_row = feature_row[h_features].copy()
            prob = float(predict_model_object(hbundle["model_object"], h_row)[0])
            summ = hbundle["training_summary"]
            rows.append({
                "Horizon": f"{horizon} year(s)",
                "Prototype probability": f"{100*prob:.1f}%",
                "Horizon-model CV AUC": f"{summ.get('patient_level_5fold_auc', float('nan')):.3f}",
                "Training eyes": summ.get("n_eyes"),
                "Positive events": summ.get("n_positive_within_horizon"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("No horizon models are available in this bundle.")


def render_baseline_calculator():
    bundle = baseline_bundle
    st.header("Baseline / First-Visit Calculator")
    st.markdown("Use this when the patient only has a **first Pentacam scan**.")
    with st.expander("Model details", expanded=False):
        st.write("Model:", bundle.get("model_name"))
        st.write("Feature set:", bundle.get("feature_set"))
        st.write("Patient-level 5-fold CV AUC:", round(bundle["training_summary"].get("patient_level_5fold_auc"), 3))
        st.write("Full-fit AUC:", round(bundle["training_summary"].get("full_fit_auc"), 3))
        st.write("Training eyes:", bundle["training_summary"].get("n_eyes"))
        st.write("CXL/progression-like eyes:", bundle["training_summary"].get("n_cxl"))
        st.write("Control eyes:", bundle["training_summary"].get("n_control"))

    age = st.number_input("Patient age at baseline scan", min_value=1.0, max_value=100.0, value=25.0, step=1.0, key="baseline_age")
    st.subheader("Baseline Pentacam values")
    cols = st.columns(3)
    with cols[0]:
        A = st.number_input("Baseline A", value=7.50, step=0.01, format="%.2f", key="baseline_A")
        Kmax = st.number_input("Baseline Kmax", value=50.0, step=0.1, format="%.1f", key="baseline_Kmax")
    with cols[1]:
        B = st.number_input("Baseline B", value=6.20, step=0.01, format="%.2f", key="baseline_B")
        BAD_D = st.number_input("Baseline BAD-D", value=5.0, step=0.1, format="%.1f", key="baseline_BAD_D")
    with cols[2]:
        C = st.number_input("Baseline C / thinnest pachy", value=470.0, step=1.0, format="%.1f", key="baseline_C")
        ARTmax = st.number_input("Baseline ARTmax", value=300.0, step=1.0, format="%.1f", key="baseline_ARTmax")

    values = {"baseline_A": A, "baseline_B": B, "baseline_C": C, "baseline_Kmax": Kmax, "baseline_BAD_D": BAD_D, "baseline_ARTmax": ARTmax, "age": age}
    feature_row = pd.DataFrame([{f: values[f] for f in bundle["features"]}])
    st.subheader("Model inputs")
    shown = pd.DataFrame({"Feature": [bundle["feature_display_names"].get(f, f) for f in bundle["features"]], "Value": [float(feature_row.iloc[0][f]) for f in bundle["features"]]})
    st.dataframe(shown, width="stretch")
    show_range_warnings(bundle, feature_row)
    probability = main_predict(bundle, feature_row)
    score = 100 * probability
    category, category_text = risk_category(bundle, probability)
    st.metric("Baseline KCN progression-like score", f"{score:.1f} / 100")
    st.write("Risk category:", f"**{category}** — {category_text}")
    st.info("Research prototype only. Not externally validated. Not for clinical decision-making.")
    show_explanation_panel(bundle, feature_row)
    show_horizon_panel(bundle, feature_row)


def render_two_scan_calculator():
    bundle = two_scan_bundle
    st.header("Two-Scan Longitudinal Calculator")
    st.markdown("Use this when a patient has **two Pentacam scans**. The app automatically calculates annualized worsening.")
    with st.expander("Model details", expanded=False):
        st.write("Model:", bundle.get("model_kind", "Random Forest"))
        st.write("Feature set:", bundle.get("feature_set"))
        st.write("Patient-level 5-fold CV AUC:", round(bundle["training_summary"].get("patient_level_5fold_oof_auc"), 3))
        st.write("Full-fit AUC:", round(bundle["training_summary"].get("full_fit_auc"), 3))
        st.write("Training eyes:", bundle["training_summary"].get("n_eyes"))
        st.write("CXL/progression-like eyes:", bundle["training_summary"].get("n_cxl"))
        st.write("Control eyes:", bundle["training_summary"].get("n_control"))

    age = st.number_input("Patient age at first scan", min_value=1.0, max_value=100.0, value=25.0, step=1.0, key="twoscan_age")
    years = st.number_input("Years between scans", min_value=0.05, max_value=10.0, value=1.0, step=0.1, key="twoscan_years")
    st.subheader("Pentacam values")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**A / anterior radius**")
        A_first = st.number_input("A first scan", value=7.50, step=0.01, format="%.2f", key="A_first")
        A_final = st.number_input("A final/pre-CXL scan", value=7.40, step=0.01, format="%.2f", key="A_final")
    with cols[1]:
        st.markdown("**B / posterior radius**")
        B_first = st.number_input("B first scan", value=6.20, step=0.01, format="%.2f", key="B_first")
        B_final = st.number_input("B final/pre-CXL scan", value=6.10, step=0.01, format="%.2f", key="B_final")
    with cols[2]:
        st.markdown("**C / thinnest pachy**")
        C_first = st.number_input("C first scan", value=470.0, step=1.0, format="%.1f", key="C_first")
        C_final = st.number_input("C final/pre-CXL scan", value=462.0, step=1.0, format="%.1f", key="C_final")
    cols2 = st.columns(3)
    with cols2[0]:
        st.markdown("**Kmax**")
        K_first = st.number_input("Kmax first scan", value=50.0, step=0.1, format="%.1f", key="K_first")
        K_final = st.number_input("Kmax final/pre-CXL scan", value=51.0, step=0.1, format="%.1f", key="K_final")
    with cols2[1]:
        st.markdown("**BAD-D**")
        BAD_first = st.number_input("BAD-D first scan", value=5.0, step=0.1, format="%.1f", key="BAD_first")
        BAD_final = st.number_input("BAD-D final/pre-CXL scan", value=5.5, step=0.1, format="%.1f", key="BAD_final")
    with cols2[2]:
        st.markdown("**ARTmax**")
        ART_first = st.number_input("ARTmax first scan", value=300.0, step=1.0, format="%.1f", key="ART_first")
        ART_final = st.number_input("ARTmax final/pre-CXL scan", value=285.0, step=1.0, format="%.1f", key="ART_final")

    def worsening(first, final, years, decrease_is_worse):
        slope = (final - first) / years
        return -slope if decrease_is_worse else slope

    values = {
        "baseline_A": A_first, "baseline_B": B_first, "baseline_C": C_first,
        "baseline_Kmax": K_first, "baseline_BAD_D": BAD_first, "baseline_ARTmax": ART_first,
        "A_worse_per_year": worsening(A_first, A_final, years, True),
        "B_worse_per_year": worsening(B_first, B_final, years, True),
        "C_worse_per_year": worsening(C_first, C_final, years, True),
        "Kmax_worse_per_year": worsening(K_first, K_final, years, False),
        "BAD_D_worse_per_year": worsening(BAD_first, BAD_final, years, False),
        "ARTmax_worse_per_year": worsening(ART_first, ART_final, years, True),
        "age": age,
    }
    feature_row = pd.DataFrame([{f: values[f] for f in bundle["features"]}])
    st.subheader("Auto-calculated model inputs")
    shown = pd.DataFrame({"Feature": [bundle["feature_display_names"].get(f, f) for f in bundle["features"]], "Value": [float(feature_row.iloc[0][f]) for f in bundle["features"]]})
    st.dataframe(shown, width="stretch")
    show_range_warnings(bundle, feature_row)
    probability = main_predict(bundle, feature_row)
    score = 100 * probability
    category, category_text = risk_category(bundle, probability)
    st.metric("KCN progression-like score", f"{score:.1f} / 100")
    st.write("Risk category:", f"**{category}** — {category_text}")
    st.info("Research prototype only. Not externally validated. Not for clinical decision-making.")
    show_explanation_panel(bundle, feature_row)
    show_horizon_panel(bundle, feature_row)


st.title("KCN Progression-Like Risk Calculators")
st.caption("Research prototype — not for clinical decision-making")
st.warning("These calculators estimate similarity to CXL/progression-like eyes in the training cohort. They are not externally validated and should not be used as clinical decision tools.")

tab1, tab2 = st.tabs(["Baseline / first visit", "Two-scan longitudinal"])
with tab1:
    render_baseline_calculator()
with tab2:
    render_two_scan_calculator()
