import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from dowhy import CausalModel
from econml.dr import DRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def calculate_ate_with_psm(df, treatment_col, outcome_col, confounder_cols):
    ps_model = LogisticRegression()
    ps_model.fit(df[confounder_cols], df[treatment_col])
    df['Propensity_Score'] = ps_model.predict_proba(df[confounder_cols])[:, 1]
    
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['Propensity_Score']])
    distances, indices = nn.kneighbors(treated[['Propensity_Score']])
    
    matched_control = control.iloc[indices.flatten()]
    ate = treated[outcome_col].mean() - matched_control[outcome_col].mean()
    
    return ate, df

def run_dowhy_analysis(df, treatment, outcome, graph):
    model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph)
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_matching",
        target_units="ate"
    )
    return model, identified_estimand, estimate

def run_uplift_modeling(df, treatment_col, outcome_col, feature_cols):
    X = df[feature_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    est = DRLearner(
        model_regression=GradientBoostingRegressor(),
        model_propensity=GradientBoostingClassifier(),
        model_final=GradientBoostingRegressor(),
        cv=3
    )
    
    est.fit(Y, T, X=X)
    uplift_scores = est.effect(X)
    return est, uplift_scores
