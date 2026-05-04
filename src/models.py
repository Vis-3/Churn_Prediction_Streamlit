from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train))
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'model': model,
            'roc_auc': roc_auc_score(y_test, y_prob),
            'report': classification_report(y_test, y_pred)
        }
    return results

def train_smote_xgb(X_train, y_train, X_test, y_test):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_res, y_res)
    return xgb, X_res, y_res

def run_shap_analysis(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values
