"""
Loan Approval Prediction  (MCA109P Project)
Team: Saneha (2026010103), Siddharth (2026010112)

Goal: predict whether a loan will be fully repaid (0) or NOT fully repaid (1)
so that a lender can make faster, more consistent approval decisions.

Run:  python loan_approval_prediction.py
Needs: loan_data.csv in the same folder, and pandas, numpy, scikit-learn,
       matplotlib, seaborn
Output: figures/*.png  and  results.json
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
NAVY, ORANGE = "#1f3a6e", "#e07b39"

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("loan_data.csv")
target = "not.fully.paid"
results = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_total": int(df.isna().sum().sum()),
    "duplicates": int(df.duplicated().sum()),
    "class_counts": {int(k): int(v) for k, v in df[target].value_counts().items()},
}
results["majority_baseline_accuracy"] = float(1 - df[target].mean())

# ---------------------------------------------------------------- 2. EDA
# 2.1 class balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = df[target].value_counts().sort_index()
ax.bar(["Fully paid (0)", "Not fully paid (1)"], counts.values, color=[NAVY, ORANGE])
for i, v in enumerate(counts.values):
    ax.text(i, v + 80, f"{v}\n({v / len(df):.1%})", ha="center")
ax.set_ylabel("Number of loans")
ax.set_title("Class balance of the target")
ax.set_ylim(0, counts.max() * 1.18)
fig.tight_layout(); fig.savefig("figures/01_class_balance.png", dpi=160); plt.close(fig)

# 2.2 loan purpose: volume and not-fully-paid rate
purpose = df.groupby("purpose")[target].agg(["count", "mean"]).sort_values("count", ascending=False)
results["purpose_table"] = {k: {"count": int(r["count"]), "rate": float(r["mean"])} for k, r in purpose.iterrows()}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].barh(purpose.index[::-1], purpose["count"][::-1], color=NAVY)
axes[0].set_title("Loans by purpose"); axes[0].set_xlabel("Count")
axes[1].barh(purpose.index[::-1], purpose["mean"][::-1] * 100, color=ORANGE)
axes[1].axvline(df[target].mean() * 100, color="k", ls="--", lw=1, label="Overall rate")
axes[1].set_title("Not-fully-paid rate by purpose"); axes[1].set_xlabel("%"); axes[1].legend()
fig.tight_layout(); fig.savefig("figures/02_purpose.png", dpi=160); plt.close(fig)

# 2.3 FICO and interest rate vs outcome
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for val, col, lab in [(0, NAVY, "Fully paid"), (1, ORANGE, "Not fully paid")]:
    axes[0].hist(df.loc[df[target] == val, "fico"], bins=30, alpha=0.65, color=col, label=lab, density=True)
axes[0].set_title("FICO score distribution"); axes[0].set_xlabel("FICO"); axes[0].legend()
sns.boxplot(data=df, x=target, y="int.rate", ax=axes[1], palette=[NAVY, ORANGE])
axes[1].set_xticklabels(["Fully paid", "Not fully paid"]); axes[1].set_xlabel("")
axes[1].set_title("Interest rate by outcome")
fig.tight_layout(); fig.savefig("figures/03_fico_intrate.png", dpi=160); plt.close(fig)

# 2.4 credit policy and inquiries
cp = df.groupby("credit.policy")[target].mean() * 100
inq = df.assign(inq=df["inq.last.6mths"].clip(upper=8)).groupby("inq")[target].mean() * 100
results["credit_policy_rate"] = {int(k): float(v) for k, v in cp.items()}
results["inq_rate"] = {int(k): float(v) for k, v in inq.items()}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(["Policy not met (0)", "Policy met (1)"], cp.values, color=[ORANGE, NAVY])
axes[0].set_ylabel("Not-fully-paid rate (%)"); axes[0].set_title("Credit policy vs outcome")
axes[1].plot(inq.index, inq.values, marker="o", color=NAVY)
axes[1].set_xlabel("Inquiries in last 6 months (8 = 8 or more)")
axes[1].set_ylabel("Not-fully-paid rate (%)"); axes[1].set_title("Recent credit inquiries vs outcome")
fig.tight_layout(); fig.savefig("figures/04_policy_inquiries.png", dpi=160); plt.close(fig)

# 2.5 correlation heatmap
num = df.drop(columns="purpose")
corr = num.corr()
results["corr_with_target"] = {k: float(v) for k, v in corr[target].drop(target).sort_values(key=abs, ascending=False).items()}
fig, ax = plt.subplots(figsize=(9, 7.2))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7}, cbar_kws={"shrink": .8})
ax.set_title("Correlation matrix")
fig.tight_layout(); fig.savefig("figures/05_correlation.png", dpi=160); plt.close(fig)

# 2.6 numeric distributions
cols = ["int.rate", "installment", "log.annual.inc", "dti", "days.with.cr.line", "revol.util"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, c in zip(axes.ravel(), cols):
    ax.hist(df[c], bins=30, color=NAVY); ax.set_title(c)
fig.suptitle("Distribution of key numeric features")
fig.tight_layout(); fig.savefig("figures/06_distributions.png", dpi=160); plt.close(fig)

results["describe"] = json.loads(df.drop(columns="purpose").describe().T.round(4).to_json(orient="index"))

# ---------------------------------------------------------------- 3. PREPARE
X = pd.get_dummies(df.drop(columns=target), columns=["purpose"], drop_first=True).astype(float)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
results["split"] = {"train": int(len(X_train)), "test": int(len(X_test)),
                    "n_features_after_encoding": int(X.shape[1]),
                    "train_positive_rate": float(y_train.mean()),
                    "test_positive_rate": float(y_test.mean())}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- 4. MODELS
models = {}
best_params = {}

lr = Pipeline([("scale", StandardScaler()),
               ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))])
lr.fit(X_train, y_train); models["Logistic Regression"] = lr

dt_grid = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                       {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [20, 50, 100]},
                       scoring="roc_auc", cv=cv, n_jobs=-1)
dt_grid.fit(X_train, y_train); models["Decision Tree"] = dt_grid.best_estimator_
best_params["Decision Tree"] = dt_grid.best_params_

rf_grid = GridSearchCV(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                              random_state=RANDOM_STATE, n_jobs=-1),
                       {"max_depth": [6, 10, 14], "min_samples_leaf": [5, 10, 20]},
                       scoring="roc_auc", cv=cv, n_jobs=1)
rf_grid.fit(X_train, y_train); models["Random Forest"] = rf_grid.best_estimator_
best_params["Random Forest"] = rf_grid.best_params_

# extra experiment: random forest WITHOUT class weights (to show why accuracy misleads)
rf_plain = RandomForestClassifier(n_estimators=300, max_depth=rf_grid.best_params_["max_depth"],
                                  min_samples_leaf=rf_grid.best_params_["min_samples_leaf"],
                                  random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
models["Random Forest (no class weights)"] = rf_plain
results["best_params"] = {k: {a: int(b) for a, b in v.items()} for k, v in best_params.items()}

# ---------------------------------------------------------------- 5. EVALUATE
metrics, cms, roc = {}, {}, {}
for name, m in models.items():
    proba = m.predict_proba(X_test)[:, 1]
    pred = m.predict(X_test)
    cvauc = cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc")
    metrics[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_auc_mean": float(cvauc.mean()), "cv_auc_std": float(cvauc.std()),
    }
    cms[name] = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba); roc[name] = (fpr, tpr)
results["metrics"] = {k: {a: float(b) for a, b in v.items()} for k, v in metrics.items()}
results["confusion"] = cms

main_models = ["Logistic Regression", "Decision Tree", "Random Forest"]

# 5.1 ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, c in zip(main_models, [NAVY, "#3a9d5d", ORANGE]):
    ax.plot(*roc[name], label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})", color=c, lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves (test set)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("figures/07_roc.png", dpi=160); plt.close(fig)

# 5.2 confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, name in zip(axes, main_models):
    sns.heatmap(np.array(cms[name]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Paid", "Not paid"], yticklabels=["Paid", "Not paid"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.tight_layout(); fig.savefig("figures/08_confusion.png", dpi=160); plt.close(fig)

# 5.3 metric comparison
mdf = pd.DataFrame(metrics).T.loc[main_models, ["accuracy", "precision", "recall", "f1", "roc_auc"]]
ax = mdf.plot(kind="bar", figsize=(9, 4.5), color=[NAVY, "#5b8bd0", ORANGE, "#3a9d5d", "#8a8a8a"], width=0.8)
ax.set_ylim(0, 1); ax.set_title("Model comparison (test set)"); ax.set_xticklabels(main_models, rotation=0)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
plt.tight_layout(); plt.savefig("figures/09_model_comparison.png", dpi=160); plt.close()

# 5.4 accuracy trap: balanced vs unweighted random forest
fig, ax = plt.subplots(figsize=(7, 4))
names = ["Always predict 'paid'", "RF (no weights)", "RF (balanced)"]
acc = [results["majority_baseline_accuracy"], metrics["Random Forest (no class weights)"]["accuracy"], metrics["Random Forest"]["accuracy"]]
rec = [0.0, metrics["Random Forest (no class weights)"]["recall"], metrics["Random Forest"]["recall"]]
x = np.arange(3)
ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color=NAVY); ax.bar(x + 0.2, rec, 0.4, label="Recall (risky loans caught)", color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Why accuracy alone is misleading")
fig.tight_layout(); fig.savefig("figures/10_accuracy_trap.png", dpi=160); plt.close(fig)

# 5.5 feature importance (RF) and LR coefficients
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
results["rf_importance"] = {k: float(v) for k, v in imp.head(10).items()}
fig, ax = plt.subplots(figsize=(7, 5))
imp.head(12)[::-1].plot(kind="barh", color=NAVY, ax=ax)
ax.set_title("Random Forest: top feature importances"); ax.set_xlabel("Importance")
fig.tight_layout(); fig.savefig("figures/11_rf_importance.png", dpi=160); plt.close(fig)

coef = pd.Series(models["Logistic Regression"].named_steps["clf"].coef_[0], index=X.columns).sort_values()
results["lr_coef"] = {k: float(v) for k, v in coef.items()}
top = pd.concat([coef.head(6), coef.tail(6)])
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.index, top.values, color=[NAVY if v < 0 else ORANGE for v in top.values])
ax.axvline(0, color="k", lw=1); ax.set_title("Logistic Regression: strongest standardized coefficients")
ax.set_xlabel("Coefficient (orange = raises risk of not repaying)")
fig.tight_layout(); fig.savefig("figures/12_lr_coef.png", dpi=160); plt.close(fig)

# 5.6 decision tree picture (top levels)
from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(models["Decision Tree"], feature_names=X.columns, class_names=["Paid", "Not paid"],
          max_depth=2, filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Decision Tree (top 3 levels)")
fig.tight_layout(); fig.savefig("figures/13_decision_tree.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 6. SAVE
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(pd.DataFrame(metrics).T.round(3).to_string())
print("Best params:", best_params)
print("Top RF features:\n", imp.head(8).round(4).to_string())
