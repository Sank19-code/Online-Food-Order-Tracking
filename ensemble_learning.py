# Ensemble Learning on a CSV / Excel dataset
# Compares: single Decision Tree vs Bagging, Random Forest, AdaBoost,
#           Gradient Boosting, Voting and Stacking
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD DATA  ---------------------------------------------------------
FILE = "diabetes_risk.csv"          # or "mydata.xlsx"
TARGET = "diabetes_risk"            # column to predict
DROP = ["patient_id"]               # ID columns (no information)

df = pd.read_csv(FILE) if FILE.endswith(".csv") else pd.read_excel(FILE)
print("Shape:", df.shape)

# 2. CLEAN: fill blanks instead of dropping rows  -----------------------
df = df.drop(columns=DROP, errors="ignore")
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())   # number -> median
    else:
        df[col] = df[col].fillna("Unknown")          # text -> new category

# 3. FEATURES / TARGET + ENCODE TEXT COLUMNS  ---------------------------
X = pd.get_dummies(df.drop(columns=[TARGET]))
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. DEFINE THE MODELS  -------------------------------------------------
tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)

models = {
    # Baseline: one tree (same as Lab 8)
    "Single Decision Tree": tree,

    # BAGGING: many full trees on bootstrap samples, majority vote
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(criterion="entropy", random_state=42),
        n_estimators=100, random_state=42, n_jobs=-1),

    # RANDOM FOREST: bagging + random subset of features at each split
    "Random Forest": RandomForestClassifier(
        n_estimators=200, criterion="entropy", random_state=42, n_jobs=-1),

    # BOOSTING (AdaBoost): shallow trees, each focuses on previous mistakes
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=200, learning_rate=0.5, random_state=42),

    # BOOSTING (Gradient): each tree fits the remaining error
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),

    # VOTING: different models, average their probabilities (soft voting)
    "Voting (soft)": VotingClassifier(estimators=[
        ("dt", tree),
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
    ], voting="soft", n_jobs=-1),

    # STACKING: a meta-model (LogReg) learns how to combine base models
    "Stacking": StackingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
    ], final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1),
}

# 5. TRAIN + EVALUATE EACH MODEL  ---------------------------------------
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name:22s} accuracy = {acc:.3f}")

# 6. DETAILED REPORT FOR THE BEST MODEL  --------------------------------
best_name = max(results, key=results.get)
best = models[best_name]
y_pred = best.predict(X_test)
print(f"\nBest model: {best_name}")
print(classification_report(y_test, y_pred))
print("Confusion matrix (rows = actual, cols = predicted):",
      list(best.classes_))
print(confusion_matrix(y_test, y_pred, labels=best.classes_))

# 7. FEATURE IMPORTANCE FROM THE RANDOM FOREST  -------------------------
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features (Random Forest):\n", imp.head(10))

# 8. PLOT: ACCURACY COMPARISON  -----------------------------------------
res = pd.Series(results).sort_values()
plt.figure(figsize=(8, 4.5))
bars = plt.barh(res.index, res.values, color="#4C78A8")
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.xlim(min(res.values) - 0.05, 1.0)
plt.xlabel("Test accuracy")
plt.title("Single tree vs ensemble methods")
plt.tight_layout()
plt.savefig("ensemble_comparison.png", dpi=120)
plt.show()
