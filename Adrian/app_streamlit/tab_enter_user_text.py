import streamlit as st
from sentence_transformers import SentenceTransformer

from Adrian.app_streamlit import repository
from Adrian.app_streamlit.repository import Keys
from Adrian.app_streamlit.utils import convert_answer_number_to_text


def predict(text: str) -> None:
    transformer: SentenceTransformer = repository.get_item(Keys.Transformer)
    embedded_text = transformer.encode(text, convert_to_tensor=False)
    prediction = repository.get_item(Keys.Model).predict(embedded_text.reshape(1, -1))
    repository.set_item(Keys.UserInputPrediction, convert_answer_number_to_text(prediction[0]))


def main() -> None:
    input_text = st.text_area("Enter Your text below:", placeholder="Enter Your text...")
    st.button("Check", on_click=predict, args=(input_text,))
    if repository.contains(Keys.UserInputPrediction):
        st.text(f"Result: {repository.get_item(Keys.UserInputPrediction)}")
