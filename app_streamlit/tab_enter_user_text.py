import streamlit as st
from sentence_transformers import SentenceTransformer

import repository
from repository import Key


def convert_answer_number_to_text(number: int):
    return 'FAKE' if number == 1 else 'REAL'


def predict(text: str) -> None:
    transformer: SentenceTransformer = repository.get_item(Key.Transformer)
    embedded_text = transformer.encode(text, convert_to_tensor=False)
    prediction = repository.get_item(Key.ModelSVC).predict(embedded_text.reshape(1, -1))
    repository.set_item(Key.UserInputPrediction, convert_answer_number_to_text(prediction[0]))


def main() -> None:
    input_text = st.text_area("Enter Your text below:", placeholder="Enter Your text...")
    st.button("Check", on_click=predict, args=(input_text,))
    if repository.contains(Key.UserInputPrediction):
        st.text(f"Result: {repository.get_item(Key.UserInputPrediction)}")
