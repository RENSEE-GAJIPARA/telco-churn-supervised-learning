import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📡", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(90deg,#b0001e,#e4572e);padding:1.6rem 2rem;border-radius:14px;color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.3rem 0 0;opacity:.92}
.card{padding:1.2rem;border-radius:12px;text-align:center;color:white;font-size:1.1rem}
.good{background:#2e9e5b}.bad{background:#d9534f}.mid{background:#e8a33d}.info{background:#4C78A8}
.card b{font-size:2rem;display:block}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📡 Customer Churn Predictor</h1>'
            '<p>Telco subscriber churn prediction - multi-algorithm classification (KNN, Naive Bayes, SVM, Decision Tree) with SMOTE</p></div>',
            unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load():
    """Load the pickled pipeline + metadata created by the notebook (Models/ folder, or next to app.py)."""
    for folder in (os.path.join(BASE, "models"), BASE):
        pkl, js = os.path.join(folder, "churn_model.pkl"), os.path.join(folder, "model_info.json")
        if os.path.exists(pkl) and os.path.exists(js):
            with open(pkl, "rb") as f:
                pipe = pickle.load(f)
            with open(js) as f:
                return pipe, json.load(f)
    raise FileNotFoundError


try:
    pipe, info = load()
except FileNotFoundError:
    st.error("Model files not found. Run CustomerChurn_SupervisedLearning.ipynb first so it creates the Models/ folder "
            "(churn_model.pkl + model_info.json), then restart this app.")
    st.stop()

# ----------------------------------------------------------------------------------------------
# Same feature preparation as prepare() in the notebook (the pickle holds one-hot encoding + scaling + model)
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
BINARY_YES = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines"]
TENURE_ORDER = {"New": 0, "Mid": 1, "Senior": 2, "Loyal": 3}
RAW_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService",
            *ADDONS, "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy().drop(columns=["customerID", "Churn"], errors="ignore")
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(info["tc_median"])
    grp = pd.cut(d["tenure"], bins=[-1, 12, 36, 60, np.inf], labels=list(TENURE_ORDER)).astype(str)
    d["tenure_group"] = grp.map(TENURE_ORDER).astype(int)
    d["num_services"] = (d[ADDONS] == "Yes").sum(axis=1)
    d["AutoPay"] = d["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    for c in BINARY_YES:
        d[c] = (d[c] == "Yes").astype(int)
    d["gender"] = (d["gender"] == "Male").astype(int)
    return d.drop(columns=ADDONS)[info["features"]]


def score(raw: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(prepare(raw))[:, 1]


def risk_band(p, thr):
    if p >= max(thr, 0.7):
        return "High risk", "bad"
    if p >= thr:
        return "Elevated risk", "mid"
    return "Low risk", "good"


with st.sidebar:
    st.header("⚙️ Prediction Settings")
    thr = st.slider(
        "Decision threshold", 0.10, 0.90, float(info["threshold"]), 0.05,
        help="Customers at or above this churn probability are flagged. Lowering the threshold may catch more churners but can increase false alarms."
    )

    st.divider()
    st.subheader("📌 Model Overview")
    st.write(f"**Selected model:** {info['final_model']}")
    st.write(f"**Training customers:** {info['n_customers']:,}")
    st.write(f"**Training churn rate:** {info['churn_rate']:.1%}")

    st.divider()
    st.subheader("🧭 How to use")
    st.markdown(
        "1. Open the **Predict** tab.\n"
        "2. Enter the customer's profile and service details.\n"
        "3. Click **Predict churn**.\n"
        "4. Review the risk level and suggested actions."
    )

    st.divider()
    st.subheader("🚦 Risk guide")
    st.markdown(
        "- 🟢 **Low risk:** below the decision threshold\n"
        "- 🟠 **Elevated risk:** at/above threshold but below 70%\n"
        "- 🔴 **High risk:** 70% or higher"
    )
    st.caption("Predictions support retention planning; they are not guarantees.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# ---------------------------------------------------------------------------- single prediction
with tab1:
    opt, rng = info["options"], info["ranges"]
    with st.form("customer"):
        st.subheader("Customer profile")
        c = st.columns(4)
        gender = c[0].selectbox("Gender", ["Female", "Male"])
        senior = c[1].selectbox("Senior citizen", ["No", "Yes"])
        partner = c[2].selectbox("Partner", ["No", "Yes"])
        dependents = c[3].selectbox("Dependents", ["No", "Yes"])

        st.subheader("Account & contract")
        c = st.columns(4)
        tenure = c[0].slider("Tenure (months)", int(rng["tenure"][0]), int(rng["tenure"][1]), int(rng["tenure"][2]))
        contract = c[1].selectbox("Contract", opt["Contract"])
        paperless = c[2].selectbox("Paperless billing", ["Yes", "No"])
        payment = c[3].selectbox("Payment method", opt["PaymentMethod"])
        c = st.columns(3)
        monthly = c[0].number_input("Monthly charges (USD)", float(rng["MonthlyCharges"][0]), float(rng["MonthlyCharges"][1]),
                                    float(round(rng["MonthlyCharges"][2], 2)), step=0.5)
        auto_total = c[1].checkbox("Estimate total charges (tenure x monthly)", value=True)
        total = c[2].number_input("Total charges (USD)", 0.0, float(rng["TotalCharges"][1]) * 1.5,
                                  float(round(rng["TotalCharges"][2], 2)), step=10.0, disabled=auto_total)

        st.subheader("Services")
        c = st.columns(4)
        phone = c[0].selectbox("Phone service", ["Yes", "No"])
        lines = c[1].selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = c[2].selectbox("Internet service", opt["InternetService"])
        c = st.columns(3)
        addon_vals = {}
        for i, name in enumerate(ADDONS):
            label = {"OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
                     "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies"}[name]
            addon_vals[name] = c[i % 3].selectbox(label, ["No", "Yes"])
        go = st.form_submit_button("Predict churn", width="stretch", type="primary")

    if go:
        if internet == "No":                      # add-ons do not exist without internet
            addon_vals = {k: "No internet service" for k in addon_vals}
        if phone == "No":
            lines = "No phone service"
        row = {"gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner, "Dependents": dependents,
               "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet, **addon_vals,
               "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
               "TotalCharges": tenure * monthly if auto_total else total}
        prob = float(score(pd.DataFrame([row])[RAW_COLS])[0])
        band, css = risk_band(prob, thr)
        a, b, d = st.columns(3)
        a.markdown(f'<div class="card {css}">Churn risk<b>{band}</b></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="card info">Churn probability<b>{prob:.0%}</b></div>', unsafe_allow_html=True)
        d.markdown(f'<div class="card info">Prediction<b>{"Will churn" if prob >= thr else "Will stay"}</b></div>',
                   unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0), text=f"Churn probability: {prob:.1%} (threshold {thr:.0%})")

        tips = []
        if contract == "Month-to-month":
            tips.append("Month-to-month contract - offer a discount or perk to move the customer to a 1- or 2-year plan.")
        if tenure <= 12:
            tips.append("New customer (first year) - schedule an onboarding check-in call.")
        if internet == "Fiber optic" and monthly > 70:
            tips.append("High fibre-optic bill - review pricing or bundle security / tech-support add-ons.")
        if "automatic" not in payment.lower():
            tips.append("Not on AutoPay - encourage automatic payment with a small incentive.")
        if prob >= thr:
            st.warning("This customer is likely to leave. Suggested retention actions:")
            for t in tips or ["Reach out proactively with a personalised retention offer."]:
                st.markdown(f"- {t}")
        else:
            st.success("This customer is likely to stay. Keep service quality high.")
            if tips:
                st.caption("Risk factors to keep an eye on: " + " | ".join(tips))

# ---------------------------------------------------------------------------- performance
with tab2:
    st.subheader(f"Deployed model - {info['final_model']} (held-out test set, churn class)")
    fm = info["final_metrics"]
    cols = st.columns(6)
    for col, (k, v) in zip(cols, fm.items()):
        col.metric(k, f"{v:.3f}" if k != "Training Time (s)" else f"{v:.3f} s")
    st.markdown("**All four models (sorted by Recall)**")
    st.dataframe(pd.DataFrame(info["all_results"]).set_index("Model"), width="stretch")
    st.markdown("**Top drivers of churn (Decision Tree importance)**")
    st.bar_chart(pd.Series(info["top_features"]).sort_values())

# ---------------------------------------------------------------------------- about
with tab3:
    st.markdown("""
**Pipeline:** raw Telco customer -> feature engineering (`tenure_group`, `num_services`, `AutoPay`) -> one-hot encoding + scaling -> classifier.
The whole encoder + model chain is stored in one pickle (`Models/churn_model.pkl`).

**Models compared:** KNN, Gaussian Naive Bayes, SVM (RBF) and Decision Tree - trained on SMOTE-balanced data, evaluated on an untouched 20% test set.

**Why Recall matters:** a missed churner (False Negative) costs the full customer lifetime value, while a false alarm only costs one retention call.
Lower the threshold in the sidebar to catch more churners at the price of more false alarms.

Predictions are estimates to support retention campaigns, not final decisions.""")