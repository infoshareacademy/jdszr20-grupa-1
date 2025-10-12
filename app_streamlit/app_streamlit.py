import joblib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tab_user_vs_ai import Name
from repository import Key
import repository, config, tab_enter_user_text, tab_user_vs_ai


def load_transformer() -> None:
    if repository.contains(Key.Transformer):
        return

    repository.set_item(Key.Transformer, SentenceTransformer(config.PATH_TRANSFORMER))


def load_model() -> None:
    if not repository.contains(Key.ModelXGBoost):
        repository.set_item(Key.ModelXGBoost, joblib.load(config.PATH_MODEL_XGBOOST))

    if not repository.contains(Key.ModelSVC):
        repository.set_item(Key.ModelSVC, joblib.load(config.PATH_MODEL_SVC))


def load_datasets() -> None:
    def load_dataset(repository_key: Key, file_path: str):
        if not repository.contains(repository_key):
            df = pd.read_csv(file_path)
            df_fake = df[df[config.COLUMN_NAME_FAKE] == 1]
            df_real = df[df[config.COLUMN_NAME_FAKE] == 0]
            repository.set_item(repository_key, {"fake": df_fake, "real": df_real})

    load_dataset(Key.DatasetValidation, config.PATH_DATA_VALIDATION)
    load_dataset(Key.DatasetGPT, config.PATH_DATA_GPT)



load_transformer()
load_model()
load_datasets()

st.header("Try and test our app!")
st.text("Disclaimer:")
st.text("Please note that our model is not a fact-checker. It does not search the Internet or any database to validate facts."
        " It has been trained on a finite numer of labeled text samples to classify news as real or fake based on typical semantic and syntactic cues.")

tab_test_yourself_svc, tab_test_yourself_xgboost, tab_enter_text = st.tabs([
    "Test Yourself vs AI! (SVC)",
    "Test Yourself vs AI! (XGBoost)",
    "Enter Your text"
])

with tab_test_yourself_svc:
    tab_user_vs_ai.main(Name.SVC)

with tab_test_yourself_xgboost:
    tab_user_vs_ai.main(Name.XGBoost)

with tab_enter_text:
    tab_enter_user_text.main()

