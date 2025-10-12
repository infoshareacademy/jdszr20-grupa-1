from enum import Enum
from typing import Any
import streamlit as st


class Key(Enum):
    Transformer = "Transformer"
    ModelSVC = "ModelSVC"
    ModelXGBoost = "ModelXGBoost"
    DatasetValidation = "DatasetValidation"
    DatasetGPT = "DatasetGPT"
    DrawnEntriesSVC = "DrawnEntriesSVC"
    DrawnEntriesXGBoost = "DrawnEntriesGXBoost"
    ScoreUserSVC = "ScoreUserSVC"
    ScoreUserXGBoost = "ScoreUserXGBoost"
    ScoreModelSVC = "ScoreModelSVC"
    ScoreModelXGBoost = "ScoreModelXGBoost"
    ResultSVC = "ResultSVC"
    ResultXGBoost = "ResultXGBoost"

    UserInputPrediction = "UserInputPrediction"


def get_item(key: Key) -> Any:
    return st.session_state[key.value]


def set_item(key: Key, value: Any) -> None:
    st.session_state[key.value] = value


def set_item_safe(key: Key, value: Any) -> None:
    """
    Assigns value to the given key if the key does not exist.
    :param key:
    :param value:
    :return:
    """
    if key.value in st.session_state:
        return

    set_item(key, value)


def del_item(key: Key) -> None:
    if not contains(key):
        return

    del st.session_state[key.value]


def contains(key: Key) -> bool:
    return key.value in st.session_state

