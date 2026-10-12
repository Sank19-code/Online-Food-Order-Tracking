"""
Train and save the two final models used by the Streamlit GUI.

    python src/train_models.py

Re-creates the tuned models selected in the notebooks, on the same leak-free,
stratified 80:20 split (random_state = 42):
  * ETA      -> GradientBoostingRegressor(learning_rate=0.05, max_depth=7, n_estimators=200)
                (regression.ipynb, section 6: best tuned model, R2 0.813)
  * Delay    -> GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=200)
                (classification.ipynb, section 7: best tuned model, weighted F1 0.844)

Each file is a dict {'preprocessor', 'model', 'features', 'metrics', ...} in
the same format that classification.ipynb writes to models/best_classifier.pkl.
"""
import os
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (FEATURES, RANDOM_STATE, ROOT, build_preprocessor, clean_and_engineer,  # noqa: E402
                      get_feature_names, load_raw_data)

MODELS_DIR = os.path.join(ROOT, 'models')
REG_PATH = os.path.join(MODELS_DIR, 'best_regressor.pkl')
CLF_PATH = os.path.join(MODELS_DIR, 'best_classifier.pkl')


def split(df):
    X = df[FEATURES]
    return train_test_split(X, df[['Time_taken', 'Is_Delayed']], test_size=0.20,
                            random_state=RANDOM_STATE, stratify=df['Is_Delayed'])


def train_regressor(X_train, X_test, y_train, y_test):
    pre = build_preprocessor()
    Xtr, Xte = pre.fit_transform(X_train), pre.transform(X_test)          # fit on TRAIN only
    model = GradientBoostingRegressor(learning_rate=0.05, max_depth=7, n_estimators=200,
                                      random_state=RANDOM_STATE).fit(Xtr, y_train)
    p = model.predict(Xte)
    resid = y_test.values - p
    metrics = {'R2': r2_score(y_test, p), 'RMSE': float(np.sqrt(mean_squared_error(y_test, p))),
               'MAE': mean_absolute_error(y_test, p), 'within_5_min': float((np.abs(resid) <= 5).mean()),
               'resid_std': float(resid.std())}
    return {'preprocessor': pre, 'model': model, 'features': FEATURES, 'metrics': metrics,
            'feature_names': get_feature_names(pre), 'name': 'Gradient Boosting Regressor (tuned)'}


def train_classifier(X_train, X_test, y_train, y_test):
    pre = build_preprocessor()
    Xtr, Xte = pre.fit_transform(X_train), pre.transform(X_test)
    model = GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=200,
                                       random_state=RANDOM_STATE).fit(Xtr, y_train)
    p, s = model.predict(Xte), model.predict_proba(Xte)[:, 1]
    metrics = {'Accuracy': accuracy_score(y_test, p),
               'Precision': precision_score(y_test, p, average='weighted'),
               'Recall': recall_score(y_test, p, average='weighted'),
               'F1': f1_score(y_test, p, average='weighted'),
               'ROC-AUC': roc_auc_score(y_test, s)}
    return {'preprocessor': pre, 'model': model, 'features': FEATURES, 'metrics': metrics,
            'feature_names': get_feature_names(pre), 'name': 'Gradient Boosting Classifier (tuned)'}


def train_all(verbose=True):
    t0 = time.time()
    df = clean_and_engineer(load_raw_data(), verbose=verbose)
    X_train, X_test, y_train, y_test = split(df)
    reg = train_regressor(X_train, X_test, y_train['Time_taken'], y_test['Time_taken'])
    clf = train_classifier(X_train, X_test, y_train['Is_Delayed'], y_test['Is_Delayed'])
    for bundle in (reg, clf):
        bundle['sklearn_version'] = sklearn.__version__
        bundle['n_train'], bundle['n_test'] = len(X_train), len(X_test)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(reg, REG_PATH, compress=3)
    joblib.dump(clf, CLF_PATH, compress=3)
    if verbose:
        print('ETA regressor   :', {k: round(v, 4) for k, v in reg['metrics'].items()})
        print('Delay classifier:', {k: round(v, 4) for k, v in clf['metrics'].items()})
        print(f'Saved {REG_PATH} and {CLF_PATH} in {time.time() - t0:.0f}s')
    return reg, clf


if __name__ == '__main__':
    train_all()
