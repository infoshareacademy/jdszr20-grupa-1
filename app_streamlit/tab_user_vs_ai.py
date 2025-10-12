from enum import Enum
from typing import List, Dict

import pandas as pd
import streamlit as st
from pandas import DataFrame

import repository, config
from repository import Key
from utils import Result


class Name(Enum):
    SVC = "SVC"
    XGBoost = "XGBoost"


class KeyName(Enum):
    Dataset = "Dataset"
    DrawnEntries = "DrawnEntries"
    ScoreUser = "ScoreUser"
    ScoreModel = "ScoreModel"
    Result = "Result"


keys_lookup: Dict[Name, Dict[KeyName, Key]] = {
    Name.SVC: {
        KeyName.Dataset: Key.DatasetValidation,
        KeyName.DrawnEntries: Key.DrawnEntriesSVC,
        KeyName.ScoreUser: Key.ScoreUserSVC,
        KeyName.ScoreModel: Key.ScoreModelSVC,
        KeyName.Result: Key.ResultSVC,
    },
    Name.XGBoost: {
        KeyName.Dataset: Key.DatasetGPT,
        KeyName.DrawnEntries: Key.DrawnEntriesXGBoost,
        KeyName.ScoreUser: Key.ScoreUserXGBoost,
        KeyName.ScoreModel: Key.ScoreModelXGBoost,
        KeyName.Result: Key.ResultXGBoost,
    }
}


def draw_and_save_entries(
        keys: Dict[KeyName, Key],
        count_fake: int,
        count_real: int
) -> None:
    if repository.contains(keys[KeyName.DrawnEntries]):
        return

    dataset = repository.get_item(keys[KeyName.Dataset])
    df_fake = dataset["fake"]
    df_real = dataset["real"]
    drawn_entries = pd.concat(
        [df_fake.sample(count_fake), df_real.sample(count_real)],
        ignore_index=True
    )

    repository.set_item(
        keys[KeyName.DrawnEntries],
        drawn_entries.sample(frac=1).reset_index(drop=True)
    )


def check_result(keys: Dict[KeyName, Key], chosen_index_user: int) -> None:
    drawn_entries: DataFrame = repository.get_item(keys[KeyName.DrawnEntries])

    entry_actual: DataFrame = drawn_entries.loc[drawn_entries[config.COLUMN_NAME_FAKE] == 1]
    chosen_index_model = drawn_entries[config.COLUMN_NAME_PROBA].idxmax()

    result = Result(
        chosen_index_user,
        chosen_index_model,
        entry_actual.index[0]
    )

    repository.set_item(keys[KeyName.Result], result)
    calculate_score(keys, result)


def calculate_score(keys: Dict[KeyName, Key], result: Result) -> None:
    if result.is_user_right:
        repository.set_item(
            keys[KeyName.ScoreUser],
            repository.get_item(keys[KeyName.ScoreUser]) + 1
        )
    if result.is_model_right:
        repository.set_item(
            keys[KeyName.ScoreModel],
            repository.get_item(keys[KeyName.ScoreModel]) + 1
        )


def load_next_set(keys: Dict[KeyName, Key]) -> None:
    repository.del_item(keys[KeyName.Result])
    repository.del_item(keys[KeyName.DrawnEntries])




def create_structure_result(result: Result | None) -> None:
    container = st.container(border=True)
    with container:
        st.header("Result")
        if result is not None:
                st.table(pd.DataFrame({
                    "User": ["OK" if result.is_user_right else "Wrong"],
                    "Model": ["OK" if result.is_model_right else "Wrong"]
                }))
        else:
            st.text("Choose Your answer to see the result.")


def create_structure_drawn_entries(
        name: Name,
        entries: List[str],
        result: Result | None
) -> None:
    for i, entry in enumerate(entries):
        container = st.container(key=f"{name.value}-container-id-{i}")
        with container:

            col_buttons, col_result = st.columns((7, 3))

            with col_buttons:
                st.button(
                    entry,
                    on_click=check_result,
                    args=(keys_lookup[name], i),
                    width="stretch",
                    disabled=repository.get_item(keys_lookup[name][KeyName.Result]) is not None,
                    type="tertiary"
                    if result is not None and
                       (i == result.index_user or
                        i == result.index_model)
                    else "secondary",
                    key=f"{name.value}-button-id-{i}"
                )

            with col_result:
                text = "<<"
                if result is not None:
                    if i == result.index_user:
                        text += " | USER"
                    if i == result.index_model:
                        text += " | MODEL"
                    if i == result.index_actual:
                        text += " | ACTUAL"

                text_placeholder = st.empty()
                if text != "<<":
                    text_placeholder.text(text)
                else:
                    text_placeholder.text("")


def create_structure_score(keys: Dict[KeyName, Key]) -> None:
    score_user, score_ai = st.columns(2)
    with score_user:
        st.header("User score")
        st.text(repository.get_item(keys[KeyName.ScoreUser]))
    with score_ai:
        st.header("AI score")
        st.text(repository.get_item(keys[KeyName.ScoreModel]))



def main(name: Name) -> None:
    keys_lookup_local = keys_lookup[name]

    draw_and_save_entries(keys_lookup_local, 1, 4)
    repository.set_item_safe(keys_lookup_local[KeyName.ScoreUser], 0)
    repository.set_item_safe(keys_lookup_local[KeyName.ScoreModel], 0)
    repository.set_item_safe(keys_lookup_local[KeyName.Result], None)

    st.text("Check whether You know better than AI which message is the fake one."
            " Below there is one fake text among the others. Can You find out which one it is?")

    create_structure_score(keys_lookup_local)
    create_structure_drawn_entries(
        name,
        list(repository.get_item(keys_lookup_local[KeyName.DrawnEntries])["text"]),
        repository.get_item(keys_lookup_local[KeyName.Result])
    )
    create_structure_result(repository.get_item(keys_lookup_local[KeyName.Result]))

    st.button(f"Next set", on_click=load_next_set, args=(keys_lookup_local,), width="stretch", key=f"{name.value}-button-next")
