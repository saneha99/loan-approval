"""
Loan Repayment Risk Checker  (front-end for the MCA109P project)
Team: Saneha (2026010103), Siddharth (2026010112)

Run:
    pip install streamlit pandas numpy scikit-learn
    streamlit run app.py

Keep loan_data.csv in the same folder as this file.
The model (Logistic Regression, the best model in our report) is trained
automatically the first time the app starts, so no extra model file is needed.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Loan Risk Checker", page_icon="🏦", layout="wide")

TARGET = "not.fully.paid"
PURPOSES = ["all_other", "credit_card", "debt_consolidation", "educational",
            "home_improvement", "major_purchase", "small_business"]


@st.cache_resource(show_spinner="Training the model on loan_data.csv ...")
def load_model():
    df = pd.read_csv("loan_data.csv")
    X = pd.get_dummies(df.drop(columns=TARGET), columns=["purpose"], drop_first=True).astype(float)
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))])
    model.fit(X_tr, y_tr)
    stats = {
        "auc": roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]),
        "recall": recall_score(y_te, model.predict(X_te)),
        "rows": len(df),
        "base_rate": float(y.mean()),
    }
    return model, list(X.columns), df, stats


model, columns, df, stats = load_model()

# ------------------------------------------------------------------ header
st.title("🏦 Loan Repayment Risk Checker")
st.caption("MCA109P project: predicts whether a loan is likely to be fully repaid, using Logistic Regression "
           f"trained on {stats['rows']:,} loans.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About the model")
    st.metric("ROC-AUC (test set)", f"{stats['auc']:.3f}")
    st.metric("Risky loans caught (recall)", f"{stats['recall']:.1%}")
    st.write("This is a **screening tool**. A high score means the application should get a closer manual "
             "review. It is not an automatic rejection.")
    st.caption("Educational project. Do not use for real lending decisions.")

# ------------------------------------------------------------------ inputs
st.subheader("Enter loan and borrower details")
c1, c2, c3 = st.columns(3)
with c1:
    purpose = st.selectbox("Loan purpose", PURPOSES, index=PURPOSES.index("debt_consolidation"))
    int_rate = st.slider("Interest rate (%)", 6.0, 22.0, 12.0, 0.1)
    installment = st.number_input("Monthly instalment", 15.0, 950.0, 270.0, 5.0)
    annual_income = st.number_input("Annual income", 1000.0, 2_000_000.0, 56000.0, 1000.0)
    credit_policy = st.selectbox("Meets lender's credit policy?", [1, 0], format_func=lambda v: "Yes" if v == 1 else "No")
with c2:
    fico = st.slider("FICO credit score", 600, 850, 707)
    dti = st.slider("Debt-to-income ratio", 0.0, 30.0, 12.7, 0.1)
    revol_util = st.slider("Revolving utilisation (%)", 0.0, 120.0, 46.0, 0.5)
    revol_bal = st.number_input("Revolving balance", 0.0, 1_300_000.0, 8600.0, 500.0)
with c3:
    cr_days = st.number_input("Days with a credit line", 180.0, 18000.0, 4140.0, 30.0)
    inq = st.slider("Credit inquiries in last 6 months", 0, 15, 1)
    delinq = st.slider("Times 30+ days late in last 2 years", 0, 10, 0)
    pub_rec = st.slider("Derogatory public records", 0, 5, 0)

row = {
    "credit.policy": credit_policy, "int.rate": int_rate / 100, "installment": installment,
    "log.annual.inc": float(np.log(annual_income)), "dti": dti, "fico": fico,
    "days.with.cr.line": cr_days, "revol.bal": revol_bal, "revol.util": revol_util,
    "inq.last.6mths": inq, "delinq.2yrs": delinq, "pub.rec": pub_rec,
}
for p in PURPOSES[1:]:  # first purpose was dropped when encoding
    row["purpose_" + p] = 1.0 if purpose == p else 0.0
x = pd.DataFrame([row])[columns].astype(float)

# ------------------------------------------------------------------ prediction
score = float(model.predict_proba(x)[0, 1])
flagged = score >= 0.5

st.divider()
left, right = st.columns([1, 1.4])
with left:
    st.subheader("Result")
    if flagged:
        st.error("⚠️ **Higher risk of not being repaid.** Send for closer manual review.")
    else:
        st.success("✅ **Lower risk.** Looks likely to be repaid.")
    st.metric("Risk score", f"{score:.0%}")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption(f"Flag threshold: 50%. In our data, {stats['base_rate']:.0%} of all loans were not fully repaid. "
               "The score is class-weighted, so read it as a ranking of risk, not an exact probability.")

with right:
    st.subheader("Why? Biggest factors for this applicant")
    scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
    contrib = pd.Series(clf.coef_[0] * scaler.transform(x)[0], index=columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(6).index).iloc[::-1]
    st.bar_chart(top.rename("Effect on risk"), horizontal=True)
    st.caption("Bars to the right push the risk up; bars to the left push it down (compared with an average borrower).")

with st.expander("Show the data behind the model"):
    st.dataframe(df.head(20))
