from typing import List

import pandas as pd
import streamlit as st

from Adrian.app_streamlit import config, repository
from Adrian.app_streamlit.repository import Key
from Adrian.app_streamlit.utils import convert_answer_number_to_text, Result


def load_dataset() -> None:
    if repository.contains(Key.Dataset):
        return

    df = pd.read_csv(config.PATH_DATA)
    df_fake = df[df["fake"] == 1]
    df_real = df[df["fake"] == 0]
    repository.set_item(Key.Dataset, {"fake": df_fake, "real": df_real})


def draw_and_save_entries(count_fake: int, count_real: int) -> None:
    if repository.contains(Key.DrawnEntries):
        return

    dataset = repository.get_item(Key.Dataset)
    df_fake = dataset["fake"]
    df_real = dataset["real"]
    drawn_entries = pd.concat([df_fake.sample(count_fake), df_real.sample(count_real)], ignore_index=True)

    repository.set_item(Key.DrawnEntries, drawn_entries.sample(frac=1).reset_index(drop=True))


def check_result(chosen_index: int) -> None:
    chosen_entry = repository.get_item(Key.DrawnEntries).loc[chosen_index]

    chosen_entry_text = chosen_entry["text"]
    chosen_entry_model_prediction = chosen_entry["fake_prediction"]
    chosen_entry_real_state = chosen_entry["fake"]

    result = Result(
        chosen_index,
        chosen_entry_text,
        convert_answer_number_to_text(chosen_entry_model_prediction),
        convert_answer_number_to_text(chosen_entry_real_state),
        chosen_entry_real_state == 1,
        chosen_entry_model_prediction == chosen_entry_real_state
    )
    repository.set_item(Key.Result, result)

    calculate_score(result)


def calculate_score(result: Result) -> None:
    if result.is_user_right:
        repository.set_item(Key.ScoreUser, repository.get_item(Key.ScoreUser) + 1)
    if result.is_model_right:
        repository.set_item(Key.ScoreAI, repository.get_item(Key.ScoreAI) + 1)


def load_next_set() -> None:
    repository.del_item(Key.Result)
    repository.del_item(Key.DrawnEntries)




def create_structure_result(result: Result | None) -> None:
    container = st.container(border=True)
    with container:
        st.header("Result")
        if result is not None:
                st.table(pd.DataFrame({
                    "Model prediction": [result.model_prediction],
                    "Real answer": [result.real_answer],
                    "User": ["OK" if result.is_user_right else "Wrong"],
                    "Model": ["OK" if result.is_model_right else "Wrong"]
                }))
        else:
            st.text("Choose Your answer to see the result.")

def create_structure_drawn_entries(entries: List[str]) -> None:
    result: Result|None = None
    if repository.contains(Key.Result):
        result = repository.get_item(Key.Result)

    for i, column in enumerate(entries):
        st.button(
            entries[i],
            on_click=check_result, args=(i,),
            width="stretch",
            disabled=repository.get_item(Key.Result) is not None,
            type="tertiary" if result is not None and i == result.index else "secondary"
        )

def create_structure_score() -> None:
    score_user, score_ai = st.columns(2)
    with score_user:
        st.header("User score")
        st.text(repository.get_item(Key.ScoreUser))
    with score_ai:
        st.header("AI score")
        st.text(repository.get_item(Key.ScoreAI))



def main() -> None:
    load_dataset()
    draw_and_save_entries(1, 4)
    repository.set_item_safe(Key.ScoreUser, 0)
    repository.set_item_safe(Key.ScoreAI, 0)
    repository.set_item_safe(Key.Result, None)

    st.text("Check whether You know better than AI which message is the fake one."
            " Below there is one fake text among the others. Can You find out which one it is?")

    create_structure_score()
    create_structure_drawn_entries(list(repository.get_item(Key.DrawnEntries)["text"]))
    create_structure_result(repository.get_item(Key.Result))

    st.button(f"Next set", on_click=load_next_set, width="stretch")
