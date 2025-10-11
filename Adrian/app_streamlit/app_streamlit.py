import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from Adrian.app_streamlit.repository import Key
from Adrian.app_streamlit import tab_user_vs_ai, tab_enter_user_text, config, repository


def load_transformer() -> None:
    if repository.contains(Key.Transformer):
        return

    repository.set_item(Key.Transformer, SentenceTransformer(config.PATH_TRANSFORMER))


def load_model() -> None:
    if repository.contains(Key.Model):
        return

    repository.set_item(Key.Model, joblib.load(config.PATH_MODEL))


load_transformer()
load_model()

st.header("Try and test our app!")
st.text("Disclaimer: A simple statement of an incorrect fact is NOT considered a fake according to our model.")
tab1, tab2 = st.tabs(["Test Yourself vs AI!", "Enter Your text"])

with tab1:
    tab_user_vs_ai.main()

with tab2:
    tab_enter_user_text.main()

