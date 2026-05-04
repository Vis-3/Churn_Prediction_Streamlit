import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import shap

def plot_roc_curve(y_test, y_prob, model_name, save_path=None):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_shap_summary(shap_values, X_test, feature_names, save_path=None):
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot')
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_uplift_distribution(uplift_scores, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(uplift_scores, kde=True)
    plt.title('Treatment Effect (Uplift) Distribution')
    plt.xlabel('Estimated Reduction in Churn Probability')
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_segmentation_heatmap(df, risk_col, uplift_col, save_path=None):
    segment_counts = df.groupby([risk_col, uplift_col]).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(segment_counts, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Customer Segments: Risk vs. Uplift')
    if save_path: plt.savefig(save_path)
    plt.show()
