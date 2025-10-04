from typing import List

import joblib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


class Result:
    def __init__(
            self,
            text: str,
            model_prediction: str,
            real_state: str,
            is_user_right: bool,
            is_model_right: bool
    ):
        self.__text: str = text
        self.__model_prediction: str = model_prediction
        self.__real_answer: str = real_state
        self.__is_user_right: bool = is_user_right
        self.__is_model_right: bool = is_model_right

    @property
    def text(self) -> str: return self.__text

    @property
    def model_prediction(self) -> str: return self.__model_prediction

    @property
    def real_answer(self) -> str: return self.__real_answer

    @property
    def is_user_right(self) -> bool: return self.__is_user_right

    @property
    def is_model_right(self) -> bool: return self.__is_model_right

def convert_answer_number_to_text(number: int):
    return 'FAKE' if number == 1 else 'REAL'

# Structure:
def structure_result(result: Result | None):
    container = st.container(border=True)
    with container:
        st.header("Result")
        if result is not None:
                st.text("Your chosen fake text:")
                st.text(result.text)
                st.table(pd.DataFrame({
                    "Model prediction": [result.model_prediction],
                    "Real answer": [result.real_answer],
                    "Is user right?": ["Yes" if result.is_user_right else "No"],
                    "Is model right?": ["Yes" if result.is_model_right else "No"]
                }))
        else:
            st.write("Choose Your answer to see the result.")

def structure_drawn_entries(entries: List[str]):
    # columns = st.columns(len(entries), gap="small", vertical_alignment="top")
    # for i, column in enumerate(columns):
    #     with column:
    #         st.button(entries[i], on_click=save_answer, args=(i,), width="stretch")

    for i, column in enumerate(entries):
        st.button(entries[i], on_click=save_answer, args=(i,), width="stretch")
# =============

# Logic:
def load_transformer():
    if "transformer" not in st.session_state:
        st.session_state["transformer"] = SentenceTransformer("all-mpnet-base-v2")

def load_model():
    if "model" not in st.session_state:
        st.session_state["model"] = joblib.load("../app/models/logistic_regression_f1_trial3.pkl")

def draw_and_save_entries(count_fake: int, count_real: int):
    if "drawn_entries" in st.session_state:
        return

    df = pd.read_csv("../app/data/all_data_50_with_titles_with_prediction.csv")
    df_fake = df[df["fake"] == 1]
    df_real = df[df["fake"] == 0]
    drawn_entries = pd.concat([df_fake.sample(count_fake), df_real.sample(count_real)], ignore_index=True)

    st.session_state["drawn_entries"] = drawn_entries.sample(frac=1).reset_index(drop=True)


def save_answer(index: int):
    st.session_state["chosen_index"] = index

def reset_result():
    if "chosen_index" not in st.session_state:
        return

    del st.session_state["chosen_index"]

def reload_result():
    if "drawn_entries" not in st.session_state:
        return

    reset_result()
    del st.session_state["drawn_entries"]

def get_result() -> Result | None:
    if "chosen_index" not in st.session_state:
        return None
    else:
        chosen_index = st.session_state["chosen_index"]
        chosen_entry = st.session_state["drawn_entries"].loc[chosen_index]

        chosen_entry_text = chosen_entry["text"]
        chosen_entry_model_prediction = chosen_entry["fake_prediction"]
        chosen_entry_real_state = chosen_entry["fake"]

        return Result(
            chosen_entry_text,
            convert_answer_number_to_text(chosen_entry_model_prediction),
            convert_answer_number_to_text(chosen_entry_real_state),
            chosen_entry_real_state == 1,
            chosen_entry_model_prediction == chosen_entry_real_state
        )

def predict(text: str):
    transformer: SentenceTransformer = st.session_state["transformer"]
    embedded_text = transformer.encode(text, convert_to_tensor=False)
    prediction = st.session_state["model"].predict(embedded_text.reshape(1, -1))
    st.session_state["user_input_prediction"] = convert_answer_number_to_text(prediction[0])
# =============

load_transformer()
load_model()
draw_and_save_entries(1, 4)

# entries = st.session_state["drawn_entries"]
# st.text(entries["text"])
# st.text(entries.iloc[0])

st.header("Try and test our app!")
tab1, tab2 = st.tabs(["Test Yourself vs AI!", "Enter Your text"])

with tab1:
    st.text("Check whether You know better than AI which message is the fake one."
            " Below there is one fake text among the others. Can You find out which one it is?")
    structure_drawn_entries(list(st.session_state["drawn_entries"]["text"]))
    structure_result(get_result())

    column1, column2 = st.columns(2)

    with column1:
        st.button(f"Reset", on_click=reset_result, width="stretch")

    with column2:
        st.button(f"Reload", on_click=reload_result, width="stretch")


with tab2:
    input_text = st.text_area("Enter Your text below:", placeholder="Enter Your text...")
    st.button("Check", on_click=predict, args=(input_text,))
    if "user_input_prediction" in st.session_state:
        st.text(f"Result: {st.session_state["user_input_prediction"]}")
    # st.text("Proin orci velit, accumsan at finibus varius, fringilla ut ante. Sed blandit aliquet malesuada. Nullam sit amet orci elementum, pharetra sem eget, malesuada nisi. Cras mollis ultrices leo, at finibus libero ultrices in. Vivamus eget auctor sem. Proin condimentum pretium ligula, et ornare nisl ultricies non. Phasellus id elementum est, sed tincidunt quam.")

