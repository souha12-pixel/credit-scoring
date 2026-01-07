"""
app_credit_scoring.py
Application Streamlit pour le Credit Scoring
pip install streamlit plotly
streamlit run app_credit_scoring.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring - Prédiction de Défaut",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.title("💳 Système de Credit Scoring")
st.markdown("---")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_pipeline_breakthrough.joblib")
        return model, True
    except FileNotFoundError:
        try:
            model = joblib.load("best_pipeline_ultra.joblib")
            return model, True
        except FileNotFoundError:
            try:
                model = joblib.load("best_pipeline_optimized.joblib")
                return model, True
            except:
                return None, False

model, model_loaded = load_model()

if not model_loaded:
    st.error(" Aucun modèle trouvé. Veuillez entraîner un modèle d'abord.")
    st.stop()

st.success(" Modèle chargé avec succès !")

# Fonction pour créer les features
def create_advanced_features(data):
    """Recréé les mêmes features que l'entraînement"""
    df = data.copy()
    
    # Features de base
    df['Income_to_Loan'] = df['Income'] / (df['LoanAmount'] + 1)
    df['Credit_to_Loan'] = df['CreditScore'] / (df['LoanAmount'] + 1)
    df['MonthlyPayment'] = df['LoanAmount'] / (df['LoanTerm'] + 1)
    df['Payment_to_Income'] = df['MonthlyPayment'] / (df['Income'] + 1)
    df['Available_Income'] = df['Income'] * (1 - df['DTIRatio'])
    
    # Interactions
    df['DTI_x_Interest'] = df['DTIRatio'] * df['InterestRate']
    df['DTI_x_LoanAmount'] = df['DTIRatio'] * df['LoanAmount']
    df['Interest_x_LoanAmount'] = df['InterestRate'] * df['LoanAmount']
    df['Credit_x_Income'] = df['CreditScore'] * df['Income']
    df['Age_x_Income'] = df['Age'] * df['Income']
    df['Age_x_Credit'] = df['Age'] * df['CreditScore']
    df['Employment_x_Income'] = df['MonthsEmployed'] * df['Income']
    
    # Risk scores
    df['Risk_Score_v1'] = (df['DTIRatio'] * df['InterestRate'] * df['LoanAmount']) / (df['Income'] * df['CreditScore'] + 1)
    df['Risk_Score_v2'] = df['LoanAmount'] / (df['CreditScore'] * df['Income'] + 1)
    df['Risk_Score_v3'] = (df['InterestRate'] * df['LoanAmount']) / (df['MonthsEmployed'] + 1)
    df['Capacity_Score'] = (df['Income'] * df['CreditScore']) / (df['LoanAmount'] * df['DTIRatio'] + 1)
    df['Stability_Score'] = (df['MonthsEmployed'] * df['Income']) / (df['Age'] + 1)
    
    # Polynomiales
    df['CreditScore_squared'] = df['CreditScore'] ** 2
    df['DTIRatio_squared'] = df['DTIRatio'] ** 2
    df['InterestRate_squared'] = df['InterestRate'] ** 2
    df['Income_sqrt'] = np.sqrt(df['Income'])
    df['LoanAmount_sqrt'] = np.sqrt(df['LoanAmount'])
    
    # Log transforms
    for col in ['Income', 'LoanAmount', 'Age', 'CreditScore', 'MonthsEmployed']:
        df[f'Log_{col}'] = np.log1p(df[col])
    
    # Binning
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 60, 100], 
                              labels=['VeryYoung', 'Young', 'MiddleAge', 'Senior', 'Elder'])
    df['Credit_Tier'] = pd.cut(df['CreditScore'], 
                                bins=[0, 580, 620, 670, 740, 850],
                                labels=['VeryPoor', 'Poor', 'Fair', 'Good', 'Excellent'])
    df['Income_Tier'] = pd.cut(df['Income'], 
                                bins=[0, 30000, 50000, 75000, 100000, 1000000],
                                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    df['DTI_Category'] = pd.cut(df['DTIRatio'], 
                                 bins=[0, 0.2, 0.36, 0.43, 1.0],
                                 labels=['Low', 'Moderate', 'High', 'VeryHigh'])
    
    # Flags
    df['HighRisk_DTI'] = (df['DTIRatio'] > 0.43).astype(int)
    df['HighRisk_LowCredit'] = (df['CreditScore'] < 620).astype(int)
    df['HighRisk_Interest'] = (df['InterestRate'] > 15).astype(int)
    df['HighRisk_ShortEmployment'] = (df['MonthsEmployed'] < 12).astype(int)
    df['HighRisk_Young'] = (df['Age'] < 25).astype(int)
    df['HighRisk_LowIncome'] = (df['Income'] < 40000).astype(int)
    df['HighRisk_LargeLoan'] = (df['LoanAmount'] > 50000).astype(int)
    df['Total_Risk_Flags'] = (
        df['HighRisk_DTI'] + df['HighRisk_LowCredit'] + df['HighRisk_Interest'] +
        df['HighRisk_ShortEmployment'] + df['HighRisk_Young'] + 
        df['HighRisk_LowIncome'] + df['HighRisk_LargeLoan']
    )
    
    return df

# Sidebar pour la saisie
st.sidebar.header(" Informations du Demandeur")

# Informations personnelles
st.sidebar.subheader(" Informations Personnelles")
age = st.sidebar.slider("Âge", 18, 80, 35)
income = st.sidebar.number_input("Revenu Annuel ($)", 10000, 500000, 50000, step=5000)
months_employed = st.sidebar.slider("Mois d'Emploi", 0, 480, 60)

education = st.sidebar.selectbox("Niveau d'Éducation", 
    ["High School", "Bachelor's", "Master's", "PhD"])
employment_type = st.sidebar.selectbox("Type d'Emploi",
    ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.sidebar.selectbox("Statut Marital",
    ["Single", "Married", "Divorced"])

# Informations financières
st.sidebar.subheader(" Informations Financières")
credit_score = st.sidebar.slider("Score de Crédit", 300, 850, 650)
num_credit_lines = st.sidebar.slider("Nombre de Lignes de Crédit", 0, 20, 3)
dti_ratio = st.sidebar.slider("Ratio DTI (Debt-to-Income)", 0.0, 1.0, 0.3, 0.01)

has_mortgage = st.sidebar.selectbox("Hypothèque", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Personnes à Charge", ["Yes", "No"])
has_cosigner = st.sidebar.selectbox("Co-Signataire", ["Yes", "No"])

# Informations sur le prêt
st.sidebar.subheader("🏦 Informations sur le Prêt")
loan_amount = st.sidebar.number_input("Montant du Prêt ($)", 1000, 500000, 25000, step=1000)
interest_rate = st.sidebar.slider("Taux d'Intérêt (%)", 1.0, 30.0, 10.0, 0.1)
loan_term = st.sidebar.slider("Durée du Prêt (mois)", 12, 360, 60)
loan_purpose = st.sidebar.selectbox("Objectif du Prêt",
    ["Home", "Auto", "Education", "Business", "Other"])

# Bouton de prédiction
predict_button = st.sidebar.button(" Prédire le Risque", type="primary", use_container_width=True)

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header(" Résultats de l'Analyse")
    
    if predict_button:
        # Créer le DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'CreditScore': [credit_score],
            'MonthsEmployed': [months_employed],
            'NumCreditLines': [num_credit_lines],
            'InterestRate': [interest_rate],
            'LoanTerm': [loan_term],
            'DTIRatio': [dti_ratio],
            'Education': [education],
            'EmploymentType': [employment_type],
            'MaritalStatus': [marital_status],
            'HasMortgage': [has_mortgage],
            'HasDependents': [has_dependents],
            'LoanPurpose': [loan_purpose],
            'HasCoSigner': [has_cosigner]
        })
        
        # Créer les features avancées
        input_data = create_advanced_features(input_data)
        
        # Prédiction
        try:
            proba = model.predict_proba(input_data)[0]
            risk_proba = proba[1] * 100
            
            # Affichage du résultat principal
            if risk_proba < 20:
                risk_level = "FAIBLE "
                color = "green"
                message = "Excellent profil ! Risque de défaut très faible."
            elif risk_proba < 40:
                risk_level = "MODÉRÉ "
                color = "orange"
                message = "Profil acceptable avec quelques points d'attention."
            else:
                risk_level = "ÉLEVÉ "
                color = "red"
                message = "Profil à risque élevé. Examen approfondi nécessaire."
            
            st.markdown(f"### Risque de Défaut : **{risk_level}**")
            st.markdown(f"**Probabilité de défaut : {risk_proba:.1f}%**")
            
            # Jauge de risque
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_proba,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Score de Risque", 'font': {'size': 24}},
                delta = {'reference': 30},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': 'lightgreen'},
                        {'range': [20, 40], 'color': 'lightyellow'},
                        {'range': [40, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(message)
            
            # Analyse détaillée
            st.markdown("---")
            st.subheader("📈 Analyse Détaillée")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                monthly_payment = loan_amount / loan_term
                st.metric("Paiement Mensuel", f"${monthly_payment:.0f}")
                
            with col_b:
                payment_income_ratio = (monthly_payment / (income/12)) * 100
                st.metric("% du Revenu", f"{payment_income_ratio:.1f}%")
                
            with col_c:
                total_interest = loan_amount * (interest_rate/100) * (loan_term/12)
                st.metric("Intérêts Totaux", f"${total_interest:.0f}")
            
            # Facteurs de risque
            st.markdown("---")
            st.subheader(" Facteurs de Risque Identifiés")
            
            risk_factors = []
            if dti_ratio > 0.43:
                risk_factors.append(" Ratio DTI élevé (> 43%)")
            if credit_score < 620:
                risk_factors.append(" Score de crédit faible (< 620)")
            if interest_rate > 15:
                risk_factors.append(" Taux d'intérêt élevé")
            if months_employed < 12:
                risk_factors.append(" Durée d'emploi courte (< 1 an)")
            if age < 25:
                risk_factors.append(" Jeune emprunteur")
            if loan_amount > 50000 and income < 75000:
                risk_factors.append(" Prêt important par rapport au revenu")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.success(" Aucun facteur de risque majeur identifié")
            
            # Recommandations
            st.markdown("---")
            st.subheader(" Recommandations")
            
            if risk_proba > 40:
                st.markdown("""
                -  **Examen approfondi requis** : Vérification des documents
                -  **Garanties supplémentaires** : Envisager un co-signataire ou une garantie
                -  **Réduction du montant** : Proposer un prêt de montant inférieur
                -  **Délai de décision** : Prendre le temps d'analyser en détail
                """)
            elif risk_proba > 20:
                st.markdown("""
                -  **Approbation possible** avec conditions
                -  **Documentation complète** requise
                -  **Suivi régulier** recommandé
                """)
            else:
                st.markdown("""
                -  **Profil excellent** : Approbation recommandée
                -  **Conditions favorables** peuvent être offertes
                -  **Client à fidéliser**
                """)
                
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")
            st.info("Vérifiez que le modèle a été entraîné avec les mêmes features.")

with col2:
    st.header("📋 Résumé du Profil")
    
    if predict_button:
        st.markdown("###  Personnel")
        st.markdown(f"**Âge:** {age} ans")
        st.markdown(f"**Emploi:** {months_employed} mois")
        st.markdown(f"**Type:** {employment_type}")
        
        st.markdown("###  Financier")
        st.markdown(f"**Revenu:** ${income:,}")
        st.markdown(f"**Crédit:** {credit_score}")
        st.markdown(f"**DTI:** {dti_ratio:.1%}")
        
        st.markdown("###  Prêt")
        st.markdown(f"**Montant:** ${loan_amount:,}")
        st.markdown(f"**Taux:** {interest_rate}%")
        st.markdown(f"**Durée:** {loan_term} mois")
        st.markdown(f"**Objectif:** {loan_purpose}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
         Credit Scoring System | Développé avec Streamlit | 
         À des fins éducatives uniquement
    </div>
""", unsafe_allow_html=True)
