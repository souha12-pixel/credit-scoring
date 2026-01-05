# Credit Scoring – Streamlit App + SHAP Graphs

This repository contains:
- **Streamlit app** to run credit-scoring predictions: `app_credit_scoring.py`
- **Training / evaluation script** (includes **SHAP** summary plot generation): `train_ultra_optimized.py`
- Dataset: `Loan_default.csv`
- Trained model pipeline: `best_pipeline_ultra.joblib`

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Run the Streamlit app

```bash
streamlit run app_credit_scoring.py
```

The app will try to load `best_pipeline_ultra.joblib` (included).  
If you retrain and create another model file, update the filename inside the app if needed.

## 3) (Optional) Retrain the model + generate SHAP plot

```bash
python train_ultra_optimized.py
```

This script saves:
- `best_pipeline_ultra.joblib` (updated model)
- `shap_ultra_optimized.png` (SHAP summary plot)

## Project structure

```
credit-scoring/
  app_credit_scoring.py
  train_ultra_optimized.py
  Loan_default.csv
  best_pipeline_ultra.joblib
  requirements.txt
  README.md
```

## Notes
- The dataset and model files are < 100 MB, so they can be pushed to GitHub without Git LFS.
