import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor


REGRESSION_MODELS = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": lambda: LinearRegression(),
}

CLASSIFICATION_MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
}


def _train_regressor(model_name, X_train, y_train):
    model_fn = REGRESSION_MODELS.get(model_name)
    if model_fn is None:
        model_fn = REGRESSION_MODELS["Random Forest"]
    model = model_fn()
    model.fit(X_train, y_train)
    return model


def _train_classifier(model_name, X_train, y_train):
    model_fn = CLASSIFICATION_MODELS.get(model_name)
    if model_fn is None:
        model_fn = CLASSIFICATION_MODELS["Random Forest"]
    model = model_fn()
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim == 2:
            importances = importances.mean(axis=0)
    else:
        return pd.DataFrame(columns=['feature', 'importance'])

    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    return df_imp


def train_poverty_predictor(X_train, y_train, model_name="Random Forest"):
    model = _train_regressor(model_name, X_train, y_train)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
    return model, cv_r2


def train_unemployment_predictor(X_train, y_train, model_name="Random Forest"):
    return train_poverty_predictor(X_train, y_train, model_name)


def train_income_classifier(X_train, y_train, model_name="Random Forest"):
    model = _train_classifier(model_name, X_train, y_train)
    cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted').mean()
    return model, cv_acc, cv_f1


def bin_income(df, col='income_median'):
    bins = [0, 4000, 8000, float('inf')]
    labels = ['Low (<RM4k)', 'Middle (RM4k-8k)', 'High (>RM8k)']
    return pd.cut(df[col], bins=bins, labels=labels)


def compute_vulnerability_score(df_features, X_scaled, poverty_model, unemp_model,
                                feature_names_poverty, feature_names_unemp,
                                kmeans_labels, cluster_names):
    pov_pred = poverty_model.predict(df_features[feature_names_poverty].values)
    pov_residual = df_features['poverty'].values - pov_pred

    unemp_pred = unemp_model.predict(df_features[feature_names_unemp].values)
    unemp_residual = df_features['u_rate'].values - unemp_pred

    pov_residual_scaled = (pov_residual - np.mean(pov_residual)) / np.std(pov_residual)
    unemp_residual_scaled = (unemp_residual - np.mean(unemp_residual)) / np.std(unemp_residual)

    vulnerability = pov_residual_scaled * 0.6 + unemp_residual_scaled * 0.4

    df_scores = pd.DataFrame({
        'state': df_features['state'].values,
        'district': df_features['district'].values,
        'poverty_actual': df_features['poverty'].values,
        'poverty_predicted': pov_pred,
        'unemp_actual': df_features['u_rate'].values,
        'unemp_predicted': unemp_pred,
        'vulnerability_score': vulnerability,
        'cluster': kmeans_labels,
    })

    if cluster_names:
        df_scores['cluster_name'] = df_scores['cluster'].map(
            cluster_names
        ).fillna(df_scores['cluster'].apply(lambda x: f'Cluster {x}'))

    df_scores = df_scores.sort_values('vulnerability_score', ascending=False)
    df_scores['rank'] = range(1, len(df_scores) + 1)

    return df_scores
