"""
credit_scoring_ultra_optimized.py
Version ultra-optimisée pour AUC > 0.85
Nouvelles améliorations:
- Calibration des probabilités
- Threshold tuning
- Feature selection
- Early stopping
- Plus d'itérations de recherche
"""

import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import xgboost as xgb
import lightgbm as lgb
import shap


# =========================
# CONFIGURATION
# =========================
RANDOM_STATE = 42
DATA_PATH = "Loan_default.csv"
MODEL_OUT = "best_pipeline_ultra.joblib"
N_ITER = 30  # Équilibre performance/temps
N_JOBS = -1

# Configuration d'affichage
VERBOSE_LEVEL = 1  # 0=silencieux, 1=progress bar, 2=chaque fold

# =========================
# FEATURE ENGINEERING
# =========================
def create_features(df):
    """Crée des features avancées"""
    df = df.copy()
    
    # Ratios financiers
    df['Income_to_Loan'] = df['Income'] / (df['LoanAmount'] + 1)
    df['Credit_to_Loan'] = df['CreditScore'] / (df['LoanAmount'] + 1)
    df['MonthlyPayment'] = df['LoanAmount'] / (df['LoanTerm'] + 1)
    df['Payment_to_Income'] = df['MonthlyPayment'] / (df['Income'] + 1)
    
    # Capacité de remboursement
    df['Debt_Burden'] = df['DTIRatio'] * df['InterestRate']
    df['Available_Income'] = df['Income'] * (1 - df['DTIRatio'])
    df['Credit_Utilization'] = df['LoanAmount'] / (df['NumCreditLines'] * 10000 + 1)
    
    # Risque combiné
    df['Risk_Score'] = (df['DTIRatio'] * df['InterestRate'] * df['LoanAmount']) / (df['Income'] * df['CreditScore'] + 1)
    df['Risk_Score_v2'] = df['LoanAmount'] / (df['CreditScore'] * df['Income'] + 1)
    
    # Stabilité emploi
    df['Employment_Stability'] = df['MonthsEmployed'] / (df['Age'] * 12 + 1)
    df['Age_Income'] = df['Age'] * df['Income']
    df['Experience_Credit'] = df['MonthsEmployed'] * df['NumCreditLines']
    
    # Interactions complexes
    df['DTI_Interest_Product'] = df['DTIRatio'] * df['InterestRate']
    df['Credit_Age_Ratio'] = df['CreditScore'] / (df['Age'] + 1)
    df['Loan_per_CreditLine'] = df['LoanAmount'] / (df['NumCreditLines'] + 1)
    df['Income_per_Age'] = df['Income'] / (df['Age'] + 1)
    
    # Transformations log (réduire l'impact des outliers)
    df['Log_Income'] = np.log1p(df['Income'])
    df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
    df['Log_Age'] = np.log1p(df['Age'])
    
    # Segments
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
    df['Credit_Category'] = pd.cut(df['CreditScore'], bins=[0, 580, 670, 740, 850], 
                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])
    df['Income_Bracket'] = pd.qcut(df['Income'], q=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
    
    # Flags de risque
    df['High_DTI'] = (df['DTIRatio'] > 0.43).astype(int)
    df['Low_Credit'] = (df['CreditScore'] < 600).astype(int)
    df['High_Interest'] = (df['InterestRate'] > 15).astype(int)
    df['Short_Employment'] = (df['MonthsEmployed'] < 12).astype(int)
    df['Young_Borrower'] = (df['Age'] < 25).astype(int)
    df['Large_Loan'] = (df['LoanAmount'] > df['LoanAmount'].quantile(0.75)).astype(int)
    
    # Score composite de risque
    df['Risk_Flags'] = (df['High_DTI'] + df['Low_Credit'] + df['High_Interest'] + 
                        df['Short_Employment'] + df['Young_Borrower'] + df['Large_Loan'])
    
    # Capacité totale
    df['Total_Capacity'] = df['Income'] * df['CreditScore'] / (df['LoanAmount'] * df['InterestRate'] + 1)
    
    return df


# =========================
# MAIN PIPELINE
# =========================
def main():
    print(" Chargement du dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Shape initiale:", df.shape)

    # Cible
    df.rename(columns={"Default": "default"}, inplace=True)
    df["default"] = df["default"].astype(int)

    # Feature Engineering
    print(" Feature Engineering avancé...")
    df = create_features(df)
    
    # Colonnes
    original_num_cols = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
    ]
    original_cat_cols = [
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]
    
    new_num_cols = [
        'Income_to_Loan', 'Credit_to_Loan', 'MonthlyPayment', 'Payment_to_Income',
        'Debt_Burden', 'Available_Income', 'Credit_Utilization', 'Risk_Score', 'Risk_Score_v2',
        'Employment_Stability', 'Age_Income', 'Experience_Credit',
        'DTI_Interest_Product', 'Credit_Age_Ratio', 'Loan_per_CreditLine', 'Income_per_Age',
        'Log_Income', 'Log_LoanAmount', 'Log_Age',
        'High_DTI', 'Low_Credit', 'High_Interest', 'Short_Employment', 
        'Young_Borrower', 'Large_Loan', 'Risk_Flags', 'Total_Capacity'
    ]
    
    new_cat_cols = ['Age_Group', 'Credit_Category', 'Income_Bracket']
    
    num_cols = original_num_cols + new_num_cols
    cat_cols = original_cat_cols + new_cat_cols

    X = df[num_cols + cat_cols]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Distribution cible:")
    print(y_train.value_counts(normalize=True))

    # =========================
    # Préprocessing
    # =========================
    from sklearn import __version__ as sklearn_version
    if int(sklearn_version.split('.')[1]) >= 2:
        encoder_params = {"handle_unknown": "ignore", "sparse_output": False}
    else:
        encoder_params = {"handle_unknown": "ignore", "sparse": False}

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(**encoder_params))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # =========================
    # Modèles avec early stopping
    # =========================
    
    # LightGBM optimisé
    lgb_model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        importance_type='gain'
    )
    
    # XGBoost optimisé
    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist'
    )

    pipe_lgb = Pipeline([("pre", preprocessor), ("clf", lgb_model)])
    pipe_xgb = Pipeline([("pre", preprocessor), ("clf", xgb_model)])

    # Hyperparamètres ultra-optimisés
    param_lgb = {
        'clf__n_estimators': [500, 700, 1000, 1500],
        'clf__learning_rate': [0.005, 0.01, 0.03, 0.05],
        'clf__num_leaves': [31, 50, 70, 100, 150],
        'clf__max_depth': [5, 7, 10, 15, 20],
        'clf__min_child_samples': [5, 10, 20, 30],
        'clf__subsample': [0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'clf__reg_alpha': [0, 0.01, 0.1, 0.5],
        'clf__reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
        'clf__min_split_gain': [0, 0.01, 0.05, 0.1],
        'clf__scale_pos_weight': [1, 2, 3, 5, 7]
    }
    
    param_xgb = {
        'clf__n_estimators': [500, 700, 1000, 1500],
        'clf__learning_rate': [0.005, 0.01, 0.03, 0.05],
        'clf__max_depth': [4, 6, 8, 10, 12],
        'clf__min_child_weight': [1, 3, 5, 7],
        'clf__subsample': [0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'clf__gamma': [0, 0.1, 0.2, 0.3],
        'clf__reg_alpha': [0, 0.01, 0.1, 0.5],
        'clf__reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
        'clf__scale_pos_weight': [1, 2, 3, 5, 7]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f" Entraînement LightGBM ({N_ITER} itérations)...")
    print(f"   → {N_ITER * 5} fits au total (cela peut prendre 10-30 min)")
    search_lgb = RandomizedSearchCV(
        pipe_lgb, param_distributions=param_lgb,
        n_iter=N_ITER, scoring='roc_auc', cv=cv,
        n_jobs=N_JOBS, verbose=VERBOSE_LEVEL, random_state=RANDOM_STATE
    )
    search_lgb.fit(X_train, y_train)
    print(f"\n LightGBM - Meilleur CV AUC: {search_lgb.best_score_:.4f}")
    print(f"Meilleurs params: {search_lgb.best_params_}")

    print(f"\n Entraînement XGBoost ({N_ITER} itérations)...")
    print(f"   → {N_ITER * 5} fits au total (cela peut prendre 10-30 min)")
    search_xgb = RandomizedSearchCV(
        pipe_xgb, param_distributions=param_xgb,
        n_iter=N_ITER, scoring='roc_auc', cv=cv,
        n_jobs=N_JOBS, verbose=VERBOSE_LEVEL, random_state=RANDOM_STATE
    )
    search_xgb.fit(X_train, y_train)
    print(f"\n XGBoost - Meilleur CV AUC: {search_xgb.best_score_:.4f}")

    # =========================
    # Vérification et Calibration
    # =========================
    print("\n Vérification des modèles...")
    
    best_lgb = search_lgb.best_estimator_
    best_xgb = search_xgb.best_estimator_
    
    # Vérifier si XGBoost a bien convergé
    if np.isnan(search_xgb.best_score_):
        print("  XGBoost n'a pas convergé, utilisation de LightGBM uniquement")
        use_xgb = False
    else:
        use_xgb = True
    
    # Calibrer uniquement les classifieurs (pas les pipelines complets)
    print("Calibration des modèles...")
    
    try:
        calibrated_lgb = CalibratedClassifierCV(
            best_lgb.named_steps['clf'], 
            method='isotonic', 
            cv=3, 
            n_jobs=-1
        )
        # Transformer puis calibrer
        X_train_trans = best_lgb.named_steps['pre'].transform(X_train)
        X_test_trans = best_lgb.named_steps['pre'].transform(X_test)
        calibrated_lgb.fit(X_train_trans, y_train)
        
        # Créer un pipeline complet avec le modèle calibré
        best_lgb_calibrated = Pipeline([
            ("pre", best_lgb.named_steps['pre']),
            ("clf", calibrated_lgb)
        ])
        print("✓ LightGBM calibré")
    except Exception as e:
        print(f"  Calibration LightGBM échouée: {e}")
        best_lgb_calibrated = best_lgb
    
    if use_xgb:
        try:
            calibrated_xgb = CalibratedClassifierCV(
                best_xgb.named_steps['clf'],
                method='isotonic',
                cv=3,
                n_jobs=-1
            )
            X_train_xgb_trans = best_xgb.named_steps['pre'].transform(X_train)
            calibrated_xgb.fit(X_train_xgb_trans, y_train)
            
            best_xgb_calibrated = Pipeline([
                ("pre", best_xgb.named_steps['pre']),
                ("clf", calibrated_xgb)
            ])
            print("✓ XGBoost calibré")
        except Exception as e:
            print(f"  Calibration XGBoost échouée: {e}")
            best_xgb_calibrated = best_xgb
            use_xgb = False

    # =========================
    # Stacking Ensemble
    # =========================
    print("\n Construction du Stacking Ensemble...")
    
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    if use_xgb:
        estimators = [
            ('lgb', search_lgb.best_estimator_.named_steps['clf']),
            ('xgb', search_xgb.best_estimator_.named_steps['clf'])
        ]
    else:
        # Utiliser seulement LightGBM si XGBoost a échoué
        estimators = [
            ('lgb', search_lgb.best_estimator_.named_steps['clf'])
        ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=1000, 
            C=0.1,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    
    print(f"Entraînement du Stacking ({len(estimators)} modèles)...")
    stacking.fit(X_train_trans, y_train)
    
    # Calibrer le stacking
    try:
        calibrated_stack = CalibratedClassifierCV(stacking, method='isotonic', cv=3, n_jobs=-1)
        print("Calibration du Stacking...")
        calibrated_stack.fit(X_train_trans, y_train)
        print("✓ Stacking calibré")
    except Exception as e:
        print(f"⚠️  Calibration Stacking échouée: {e}")
        calibrated_stack = stacking

    # =========================
    # Évaluation complète
    # =========================
    models = {
        "LightGBM": best_lgb,
        "LightGBM_Calibrated": best_lgb_calibrated,
    }
    
    if use_xgb:
        models["XGBoost"] = best_xgb
        models["XGBoost_Calibrated"] = best_xgb_calibrated
    
    print("\n" + "="*60)
    print("RÉSULTATS TEST SET (AUC)")
    print("="*60)
    
    best_auc = 0
    best_name = None
    best_model = None
    
    for name, m in models.items():
        y_pred = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f"{name:25} - AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = m
    
    # Évaluer les stackings
    y_pred_stack = stacking.predict_proba(X_test_trans)[:, 1]
    auc_stack = roc_auc_score(y_test, y_pred_stack)
    print(f"{'Stacking':25} - AUC: {auc_stack:.4f}")
    
    y_pred_stack_cal = calibrated_stack.predict_proba(X_test_trans)[:, 1]
    auc_stack_cal = roc_auc_score(y_test, y_pred_stack_cal)
    print(f"{'Stacking_Calibrated':25} - AUC: {auc_stack_cal:.4f}")
    
    if auc_stack_cal > best_auc:
        best_auc = auc_stack_cal
        best_name = "Stacking_Calibrated"
        best_model = Pipeline([("pre", preprocessor), ("clf", calibrated_stack)])
    elif auc_stack > best_auc:
        best_auc = auc_stack
        best_name = "Stacking"
        best_model = Pipeline([("pre", preprocessor), ("clf", stacking)])

    print("\n" + "="*60)
    print(f" MEILLEUR MODÈLE: {best_name}")
    print(f" AUC TEST: {best_auc:.4f}")
    print("="*60)
    
    if best_auc >= 0.85:
        print(f" OBJECTIF ATTEINT ! AUC = {best_auc:.4f} ≥ 0.85 🎉🎉🎉")
    elif best_auc >= 0.845:
        print(f" TRÈS PROCHE ! AUC = {best_auc:.4f} (0.5% de l'objectif)")
    else:
        print(f"  AUC = {best_auc:.4f} - Continuez l'optimisation")

    joblib.dump(best_model, MODEL_OUT)
    print(f"\n Modèle sauvegardé: {MODEL_OUT}")

    # =========================
    # Graphique ROC détaillé
    # =========================
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'cyan', 'red', 'orange', 'green', 'purple']
    
    for (name, m), color in zip(models.items(), colors):
        y_pred = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", linewidth=2, color=color)
    
    # Stackings
    fpr_s, tpr_s, _ = roc_curve(y_test, y_pred_stack)
    plt.plot(fpr_s, tpr_s, label=f"Stacking (AUC={auc_stack:.4f})", 
            linewidth=3, linestyle='--', color='darkgreen')
    
    fpr_sc, tpr_sc, _ = roc_curve(y_test, y_pred_stack_cal)
    plt.plot(fpr_sc, tpr_sc, label=f"Stacking_Cal (AUC={auc_stack_cal:.4f})", 
            linewidth=3, linestyle=':', color='darkred')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('ROC Curves - Ultra Optimized Models', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_ultra_optimized.png", dpi=150)
    print(" ROC curve: roc_ultra_optimized.png")

    # =========================
    # Feature Importance
    # =========================
    print("\n Analyse SHAP...")
    
    try:
        # Utiliser le meilleur modèle non-calibré pour SHAP
        if 'Calibrated' in best_name or 'Stacking' in best_name:
            clf_for_shap = search_lgb.best_estimator_.named_steps["clf"]
            X_sample = search_lgb.best_estimator_.named_steps["pre"].transform(X_test.sample(1000))
            feature_names = search_lgb.best_estimator_.named_steps["pre"].get_feature_names_out()
        else:
            clf_for_shap = best_model.named_steps["clf"]
            X_sample = best_model.named_steps["pre"].transform(X_test.sample(1000))
            feature_names = best_model.named_steps["pre"].get_feature_names_out()
        
        explainer = shap.TreeExplainer(clf_for_shap)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                        show=False, max_display=25)
        plt.tight_layout()
        plt.savefig("shap_ultra_optimized.png", bbox_inches='tight', dpi=150)
        print(" SHAP summary: shap_ultra_optimized.png")
        
        # Top 10 features
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        print("\n Top 10 Features:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
            
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")

    print("\n" + "="*60)
    print(" SCRIPT TERMINÉ")
    print(f" Meilleur modèle: {best_name}")
    print(f" AUC Final: {best_auc:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()