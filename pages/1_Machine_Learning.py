import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data_loader import load_and_preprocess_data
from src.constants import (
    CLUSTER_FEATURES, LFS_EXTRA_FEATURES, AMENITY_FEATURES,
    METRIC_LABELS, INCOME_BRACKETS,
)
from src.supervised import (
    train_poverty_predictor,
    train_unemployment_predictor,
    train_income_classifier,
    get_feature_importances,
    bin_income,
    compute_vulnerability_score,
)
from src.unsupervised import run_kmeans_pipeline

st.set_page_config(
    page_title="ML Lab — District Profiler",
    page_icon="\U0001f9ea",
    layout="wide",
)

st.title("\U0001f9ea Machine Learning Lab")
st.markdown(
    "Train supervised models on district-level economic data to predict poverty rates, "
    "unemployment, and income brackets. Also explore feature importance and identify "
    "economically vulnerable districts."
)

df_all, available_years, geo_json = load_and_preprocess_data()

if df_all is None:
    st.error("Failed to load data. Please check your connection and try again.")
    st.stop()

st.sidebar.title("\U0001f9ea ML Settings")

with st.sidebar.container(border=True):
    st.markdown("**Data**")
    train_year = st.selectbox("Train Year", available_years, index=min(1, len(available_years) - 1))
    test_year = st.selectbox("Test Year", available_years, index=0)

    same_year = train_year == test_year
    if same_year:
        split_strategy = "Random (80/20)"
        st.caption("Same year selected — using random split.")
    else:
        split_strategy = "Time-based"
        st.caption("Different years — using time-based split.")

# ---------------------------------------------------------
# Prepare data and feature lists
# ---------------------------------------------------------
all_feature_candidates = CLUSTER_FEATURES + LFS_EXTRA_FEATURES + AMENITY_FEATURES
available_features = [f for f in all_feature_candidates if f in df_all.columns and df_all[f].notna().any()]

df_train = df_all[df_all['date'].dt.year == train_year].copy()
df_test = df_all[df_all['date'].dt.year == test_year].copy()

num_cols = [f for f in available_features if f in df_train.columns]
df_train = df_train.dropna(subset=num_cols)
df_test = df_test.dropna(subset=num_cols)

poverty_features = [f for f in available_features if f != 'poverty']
unemp_features = [f for f in available_features if f != 'u_rate']
income_class_features = [f for f in available_features if f not in ('income_median', 'income_mean')]

if len(df_train) < 10 or len(df_test) < 5:
    st.warning("Not enough data points for reliable ML training. Need at least 10 districts for training.")
    st.stop()

# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------
if same_year:
    df_combined = pd.concat([df_train, df_test]).drop_duplicates()
    df_train, df_test = train_test_split(df_combined, test_size=0.2, random_state=42)

X_train_pov = df_train[poverty_features].values
y_train_pov = df_train['poverty'].values
X_test_pov = df_test[poverty_features].values
y_test_pov = df_test['poverty'].values

X_train_unemp = df_train[unemp_features].values
y_train_unemp = df_train['u_rate'].values
X_test_unemp = df_test[unemp_features].values
y_test_unemp = df_test['u_rate'].values

X_train_inc = df_train[income_class_features].values
y_train_inc = bin_income(df_train, 'income_median')
X_test_inc = df_test[income_class_features].values
y_test_inc = bin_income(df_test, 'income_median')
y_train_inc = y_train_inc.dropna()
valid_inc_train = y_train_inc.index
X_train_inc = X_train_inc[valid_inc_train]

valid_inc_test = y_test_inc.dropna().index
y_test_inc = y_test_inc.loc[valid_inc_test]
X_test_inc = X_test_inc[valid_inc_test]

# ---------------------------------------------------------
# Model selector in sidebar
# ---------------------------------------------------------
with st.sidebar.container(border=True):
    st.markdown("**Model**")
    reg_model_choice = st.selectbox(
        "Regression model", ["Random Forest", "Gradient Boosting", "Linear Regression"],
        key="reg_model"
    )
    clf_model_choice = st.selectbox(
        "Classification model", ["Random Forest", "Logistic Regression"],
        key="clf_model"
    )

tab_poverty, tab_unemp, tab_income, tab_vuln, tab_feat = st.tabs([
    "\U0001f4b0 Poverty Predictor",
    "\U0001f4c9 Unemployment Predictor",
    "\U0001f3f7 Income Bracket Classifier",
    "\U0001f6e1 Vulnerability Score",
    "\U0001f50d Feature Explorer",
])

# ============================================================
# TAB 1: POVERTY PREDICTOR
# ============================================================
with tab_poverty:
    st.subheader("\U0001f4b0 Poverty Rate Predictor")
    st.markdown(f"Train year: **{train_year}**, Test year: **{test_year}**, Split: **{split_strategy}**")

    with st.spinner("Training poverty prediction model..."):
        model_pov, cv_r2 = train_poverty_predictor(
            X_train_pov, y_train_pov, reg_model_choice
        )
        y_pred_pov = model_pov.predict(X_test_pov)
        r2 = r2_score(y_test_pov, y_pred_pov)
        mae = mean_absolute_error(y_test_pov, y_pred_pov)
        rmse = np.sqrt(mean_squared_error(y_test_pov, y_pred_pov))

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("R\u00b2 Score", f"{r2:.3f}")
    col_m2.metric("MAE", f"{mae:.2f}%")
    col_m3.metric("RMSE", f"{rmse:.2f}%")
    col_m4.metric("CV R\u00b2 (5-fold)", f"{cv_r2:.3f}")

    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        st.markdown("**Actual vs Predicted Poverty Rate**")
        fig_avp = px.scatter(
            x=y_test_pov, y=y_pred_pov,
            labels={'x': 'Actual Poverty Rate (%)', 'y': 'Predicted Poverty Rate (%)'},
            template='plotly_white', height=400
        )
        max_val = max(y_test_pov.max(), y_pred_pov.max()) * 1.1
        fig_avp.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                       name='Perfect Prediction', line=dict(dash='dash', color='gray'))
        )
        fig_avp.update_layout(showlegend=False)
        st.plotly_chart(fig_avp, use_container_width=True)

    with col_p2:
        st.markdown("**Feature Importance**")
        df_imp_pov = get_feature_importances(model_pov, poverty_features)
        fig_imp_pov = px.bar(
            df_imp_pov.tail(12), x='importance', y='feature',
            orientation='h', template='plotly_white', height=400,
            labels={'importance': 'Importance', 'feature': ''}
        )
        st.plotly_chart(fig_imp_pov, use_container_width=True)

    st.markdown("**Prediction Errors — Over/Under-Predicted Districts**")
    df_errors = df_test[['state', 'district']].copy()
    df_errors['actual_poverty'] = y_test_pov
    df_errors['predicted_poverty'] = y_pred_pov
    df_errors['residual'] = df_errors['actual_poverty'] - df_errors['predicted_poverty']
    df_errors = df_errors.sort_values('residual', ascending=False)

    col_err1, col_err2 = st.columns(2)
    with col_err1:
        st.markdown("**Most Under-Predicted** (higher poverty than expected)")
        st.dataframe(
            df_errors.head(10)[['state', 'district', 'actual_poverty', 'predicted_poverty', 'residual']]
            .style.format({
                'actual_poverty': '{:.2f}%', 'predicted_poverty': '{:.2f}%', 'residual': '{:.2f}%'
            }),
            use_container_width=True
        )
    with col_err2:
        st.markdown("**Most Over-Predicted** (lower poverty than expected)")
        st.dataframe(
            df_errors.tail(10)[['state', 'district', 'actual_poverty', 'predicted_poverty', 'residual']]
            .style.format({
                'actual_poverty': '{:.2f}%', 'predicted_poverty': '{:.2f}%', 'residual': '{:.2f}%'
            }),
            use_container_width=True
        )

    csv_pov = df_errors.to_csv(index=False)
    st.download_button("\U0001f4e5 Download Poverty Predictions", csv_pov, "poverty_predictions.csv")

# ============================================================
# TAB 2: UNEMPLOYMENT PREDICTOR
# ============================================================
with tab_unemp:
    st.subheader("\U0001f4c9 Unemployment Rate Predictor")
    st.markdown(f"Train year: **{train_year}**, Test year: **{test_year}**, Split: **{split_strategy}**")

    with st.spinner("Training unemployment prediction model..."):
        model_unemp, cv_r2_u = train_unemployment_predictor(
            X_train_unemp, y_train_unemp, reg_model_choice
        )
        y_pred_unemp = model_unemp.predict(X_test_unemp)
        r2_u = r2_score(y_test_unemp, y_pred_unemp)
        mae_u = mean_absolute_error(y_test_unemp, y_pred_unemp)
        rmse_u = np.sqrt(mean_squared_error(y_test_unemp, y_pred_unemp))

    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    col_u1.metric("R\u00b2 Score", f"{r2_u:.3f}")
    col_u2.metric("MAE", f"{mae_u:.2f}%")
    col_u3.metric("RMSE", f"{rmse_u:.2f}%")
    col_u4.metric("CV R\u00b2 (5-fold)", f"{cv_r2_u:.3f}")

    col_up1, col_up2 = st.columns([3, 2])

    with col_up1:
        st.markdown("**Actual vs Predicted Unemployment Rate**")
        fig_avp_u = px.scatter(
            x=y_test_unemp, y=y_pred_unemp,
            labels={'x': 'Actual Unemployment Rate (%)', 'y': 'Predicted Unemployment Rate (%)'},
            template='plotly_white', height=400
        )
        max_val_u = max(y_test_unemp.max(), y_pred_unemp.max()) * 1.1
        fig_avp_u.add_trace(
            go.Scatter(x=[0, max_val_u], y=[0, max_val_u], mode='lines',
                       name='Perfect Prediction', line=dict(dash='dash', color='gray'))
        )
        fig_avp_u.update_layout(showlegend=False)
        st.plotly_chart(fig_avp_u, use_container_width=True)

    with col_up2:
        st.markdown("**Feature Importance**")
        df_imp_u = get_feature_importances(model_unemp, unemp_features)
        fig_imp_u = px.bar(
            df_imp_u.tail(12), x='importance', y='feature',
            orientation='h', template='plotly_white', height=400,
            labels={'importance': 'Importance', 'feature': ''}
        )
        st.plotly_chart(fig_imp_u, use_container_width=True)

    st.markdown("**Prediction Errors — Most Under/Over-Predicted Districts**")
    df_errors_u = df_test[['state', 'district']].copy()
    df_errors_u['actual_unemp'] = y_test_unemp
    df_errors_u['predicted_unemp'] = y_pred_unemp
    df_errors_u['residual'] = df_errors_u['actual_unemp'] - df_errors_u['predicted_unemp']
    df_errors_u = df_errors_u.sort_values('residual', ascending=False)

    col_eu1, col_eu2 = st.columns(2)
    with col_eu1:
        st.markdown("**Most Under-Predicted**")
        st.dataframe(
            df_errors_u.head(10)[['state', 'district', 'actual_unemp', 'predicted_unemp', 'residual']]
            .style.format({
                'actual_unemp': '{:.2f}%', 'predicted_unemp': '{:.2f}%', 'residual': '{:.2f}%'
            }),
            use_container_width=True
        )
    with col_eu2:
        st.markdown("**Most Over-Predicted**")
        st.dataframe(
            df_errors_u.tail(10)[['state', 'district', 'actual_unemp', 'predicted_unemp', 'residual']]
            .style.format({
                'actual_unemp': '{:.2f}%', 'predicted_unemp': '{:.2f}%', 'residual': '{:.2f}%'
            }),
            use_container_width=True
        )

    csv_unemp = df_errors_u.to_csv(index=False)
    st.download_button("\U0001f4e5 Download Unemployment Predictions", csv_unemp, "unemployment_predictions.csv")

# ============================================================
# TAB 3: INCOME BRACKET CLASSIFIER
# ============================================================
with tab_income:
    st.subheader("\U0001f3f7 Income Bracket Classifier")
    st.markdown(
        "Classifies districts into Low (<RM4k), Middle (RM4k-8k), or High (>RM8k) income brackets "
        "using all available features **except** income columns."
    )
    st.markdown(f"Train year: **{train_year}**, Test year: **{test_year}**, Split: **{split_strategy}**")

    if len(y_train_inc) < 10 or y_test_inc.nunique() < 2:
        st.warning("Not enough data or class diversity for classification. Try a different year split.")
    else:
        with st.spinner("Training income bracket classifier..."):
            model_inc, cv_acc, cv_f1 = train_income_classifier(
                X_train_inc, y_train_inc, clf_model_choice
            )
            y_pred_inc = model_inc.predict(X_test_inc)
            acc = accuracy_score(y_test_inc, y_pred_inc)
            f1 = f1_score(y_test_inc, y_pred_inc, average='weighted')

        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        col_i1.metric("Accuracy", f"{acc:.3f}")
        col_i2.metric("F1 (Weighted)", f"{f1:.3f}")
        col_i3.metric("CV Accuracy (5-fold)", f"{cv_acc:.3f}")
        col_i4.metric("CV F1 (5-fold)", f"{cv_f1:.3f}")

        col_ic1, col_ic2 = st.columns(2)

        with col_ic1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test_inc, y_pred_inc)
            classes = sorted(set(y_test_inc) | set(y_pred_inc))
            fig_cm = px.imshow(
                cm, x=classes, y=classes, text_auto=True,
                color_continuous_scale='Blues', template='plotly_white',
                labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_ic2:
            st.markdown("**Feature Importance**")
            df_imp_inc = get_feature_importances(model_inc, income_class_features)
            fig_imp_inc = px.bar(
                df_imp_inc.tail(12), x='importance', y='feature',
                orientation='h', template='plotly_white', height=400,
                labels={'importance': 'Importance', 'feature': ''}
            )
            st.plotly_chart(fig_imp_inc, use_container_width=True)

        st.markdown("**Misclassified Districts**")
        df_misclass = df_test[['state', 'district']].copy()
        df_misclass = df_misclass.iloc[valid_inc_test].reset_index(drop=True)
        df_misclass['actual'] = y_test_inc.values
        df_misclass['predicted'] = y_pred_inc
        df_misclass['income_median'] = df_test['income_median'].iloc[valid_inc_test].values
        misclassified = df_misclass[df_misclass['actual'] != df_misclass['predicted']]

        if len(misclassified) > 0:
            st.dataframe(
                misclassified.sort_values('income_median', ascending=True),
                use_container_width=True
            )
            st.caption(f"{len(misclassified)} districts were misclassified out of {len(df_misclass)} test districts.")
        else:
            st.success("All districts classified correctly!")

        csv_inc = df_misclass.to_csv(index=False)
        st.download_button("\U0001f4e5 Download Classifier Results", csv_inc, "income_classifier_results.csv")

# ============================================================
# TAB 4: VULNERABILITY SCORE
# ============================================================
with tab_vuln:
    st.subheader("\U0001f6e1 Economic Vulnerability Score")
    st.markdown(
        "Composite risk index combining both poverty and unemployment model residuals. "
        "Higher score = district has **worse outcomes than predicted** by its economic profile."
    )

    must_features_pov = [f for f in poverty_features if f in df_all.columns and df_all[f].notna().any()]
    must_features_unemp = [f for f in unemp_features if f in df_all.columns and df_all[f].notna().any()]

    df_latest = df_test.dropna(subset=must_features_pov).copy()

    if len(df_latest) < 5:
        st.warning("Not enough complete districts in the test set for vulnerability scoring.")
    else:
        with st.spinner("Computing vulnerability scores..."):
            pov_feat_present = [f for f in must_features_pov if f in df_latest.columns]
            unemp_feat_present = [f for f in must_features_unemp if f in df_latest.columns]

            df_kmeans, X_scaled_v, kmeans_v, _, _ = run_kmeans_pipeline(
                df_latest,
                [f for f in CLUSTER_FEATURES if f in df_latest.columns],
                k=3
            )

            df_vuln = compute_vulnerability_score(
                df_latest, X_scaled_v,
                model_pov, model_unemp,
                pov_feat_present, unemp_feat_present,
                df_kmeans['cluster'].values,
                {1: "High-Income Metro Hubs", 0: "Middle-Income Towns & Suburbs", 2: "Rural & High-Poverty Districts"}
            )

        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.metric("Most Vulnerable", f"{df_vuln.iloc[0]['district']}, {df_vuln.iloc[0]['state']}")
        with col_v2:
            st.metric("Least Vulnerable", f"{df_vuln.iloc[-1]['district']}, {df_vuln.iloc[-1]['state']}")
        with col_v3:
            st.metric("Districts Scored", len(df_vuln))

        col_vt1, col_vt2 = st.columns(2)
        with col_vt1:
            st.markdown("**Top 10 Most Vulnerable Districts**")
            st.dataframe(
                df_vuln.head(10)[['rank', 'state', 'district', 'poverty_actual', 'poverty_predicted',
                                  'unemp_actual', 'unemp_predicted', 'vulnerability_score']]
                .style.format({
                    'poverty_actual': '{:.2f}%', 'poverty_predicted': '{:.2f}%',
                    'unemp_actual': '{:.2f}%', 'unemp_predicted': '{:.2f}%',
                    'vulnerability_score': '{:.3f}'
                }),
                use_container_width=True
            )
        with col_vt2:
            st.markdown("**Top 10 Least Vulnerable Districts**")
            st.dataframe(
                df_vuln.tail(10)[['rank', 'state', 'district', 'poverty_actual', 'poverty_predicted',
                                  'unemp_actual', 'unemp_predicted', 'vulnerability_score']]
                .style.format({
                    'poverty_actual': '{:.2f}%', 'poverty_predicted': '{:.2f}%',
                    'unemp_actual': '{:.2f}%', 'unemp_predicted': '{:.2f}%',
                    'vulnerability_score': '{:.3f}'
                }),
                use_container_width=True
            )

        st.markdown("**Full Vulnerability Ranking**")
        st.dataframe(
            df_vuln[['rank', 'state', 'district', 'poverty_actual', 'poverty_predicted',
                     'unemp_actual', 'unemp_predicted', 'vulnerability_score']],
            use_container_width=True
        )

        csv_vuln = df_vuln.to_csv(index=False)
        st.download_button("\U0001f4e5 Download Vulnerability Scores", csv_vuln, "vulnerability_scores.csv")

# ============================================================
# TAB 5: FEATURE EXPLORER
# ============================================================
with tab_feat:
    st.subheader("\U0001f50d Feature Importance Explorer")
    st.markdown(
        "Compare which features drive predictions globally and by state. "
        "Select a district to see how its individual feature values influenced its prediction."
    )

    st.markdown("### Global Feature Importance Comparison")
    col_fe1, col_fe2 = st.columns(2)

    with col_fe1:
        st.markdown("**Poverty Model**")
        df_imp_pov_full = get_feature_importances(model_pov, poverty_features)
        fig_imp_full = px.bar(
            df_imp_pov_full, x='importance', y='feature',
            orientation='h', template='plotly_white', height=400,
            labels={'importance': 'Importance', 'feature': ''}
        )
        st.plotly_chart(fig_imp_full, use_container_width=True)

    with col_fe2:
        st.markdown("**Unemployment Model**")
        df_imp_u_full = get_feature_importances(model_unemp, unemp_features)
        fig_imp_u_full = px.bar(
            df_imp_u_full, x='importance', y='feature',
            orientation='h', template='plotly_white', height=400,
            labels={'importance': 'Importance', 'feature': ''}
        )
        st.plotly_chart(fig_imp_u_full, use_container_width=True)

    st.markdown("---")
    st.markdown("### State-Level Feature Importance")
    st.markdown(
        "Select a state to see which features matter most for poverty prediction "
        "within that state only. This reveals localized economic dynamics."
    )

    all_states = sorted(df_all['state'].unique())
    feat_state = st.selectbox("Select State", all_states, key="feat_state")

    df_state = df_all[
        (df_all['date'].dt.year == test_year) & (df_all['state'] == feat_state)
    ].dropna(subset=poverty_features)

    if len(df_state) >= 5:
        X_state = df_state[poverty_features].values
        y_state = df_state['poverty'].values
        model_state, _ = train_poverty_predictor(X_state, y_state, reg_model_choice)
        df_imp_state = get_feature_importances(model_state, poverty_features)

        fig_imp_state = px.bar(
            df_imp_state, x='importance', y='feature',
            orientation='h', template='plotly_white', height=350,
            labels={'importance': 'Importance', 'feature': ''},
            title=f'Poverty Feature Importance — {feat_state}'
        )
        st.plotly_chart(fig_imp_state, use_container_width=True)
    else:
        st.info(f"Not enough districts in {feat_state} for a state-level model (found {len(df_state)}).")

    st.markdown("---")
    st.markdown("### Explain My District")
    st.markdown(
        "Pick a district to see which of its feature values pushed its poverty prediction "
        "higher or lower relative to the average."
    )

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        exp_state = st.selectbox("State", all_states, key="exp_state")
    districts_in_exp = sorted(
        df_all[(df_all['date'].dt.year == test_year) & (df_all['state'] == exp_state)]['district'].unique()
    )
    with col_ex2:
        exp_district = st.selectbox("District", districts_in_exp, key="exp_district")

    if exp_district:
        idx = df_test[(df_test['state'] == exp_state) & (df_test['district'] == exp_district)].index
        if len(idx) > 0:
            row = df_test.loc[idx[0]]
            pred = model_pov.predict([row[poverty_features].values])[0]
            actual = row['poverty']

            feature_vals = []
            for f in poverty_features:
                avg = df_test[f].mean()
                val = row[f]
                feature_vals.append({
                    'feature': METRIC_LABELS.get(f, f),
                    'value': val,
                    'avg': avg,
                    'diff_from_avg': val - avg
                })

            df_explain = pd.DataFrame(feature_vals).sort_values('diff_from_avg', ascending=False)

            st.metric(
                label=f"Poverty Rate for {exp_district}, {exp_state}",
                value=f"{actual:.2f}%",
                delta=f"{actual - pred:+.2f}% vs predicted ({pred:.2f}%)",
                delta_color="inverse"
            )

            st.markdown("**Feature values compared to district average:**")
            fig_explain = px.bar(
                df_explain, x='diff_from_avg', y='feature',
                orientation='h', template='plotly_white', height=400,
                labels={'diff_from_avg': 'Deviation from Average', 'feature': ''},
                color='diff_from_avg', color_continuous_scale='RdBu_r',
            )
            st.plotly_chart(fig_explain, use_container_width=True)
        else:
            st.info(f"{exp_district} not found in the test year data.")
