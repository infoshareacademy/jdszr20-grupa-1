import os

current_dir = os.path.dirname(__file__)

# Transformer
PATH_TRANSFORMER: str = os.path.join(current_dir, "models", "all-mpnet-base-v2")

# Models
PATH_MODEL_XGBOOST: str = os.path.join(current_dir, "models", "xgb_model_3_150.pkl")
PATH_MODEL_SVC: str = os.path.join(current_dir, "models", "svc_model_3_100.pkl")

# Data
PATH_DATA_VALIDATION: str = os.path.join(current_dir, "data", "50_words_VALIDATION_B_with_predictions_SVC.csv")
PATH_DATA_GPT: str = os.path.join(current_dir, "data", "ChatGPT_validation_300_combined_curated_with_probs.csv")
COLUMN_NAME_TEXT: str = "text"
COLUMN_NAME_FAKE: str = "fake"
COLUMN_NAME_PROBA: str = "y_proba"
COLUMN_NAME_PRED: str = "y_pred"