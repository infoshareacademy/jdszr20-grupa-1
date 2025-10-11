from enum import Enum
from typing import Any
import streamlit as st


class Keys(Enum):
    Transformer = "Transformer"
    Model = "Model"
    DrawnEntries = "DrawnEntries"
    UserInputPrediction = "UserInputPrediction"
    ScoreUser = "ScoreUser"
    ScoreAI = "ScoreAI"
    Result = "Result"


def get_item(key: Keys) -> Any:
    return st.session_state[key.value]


def set_item(key: Keys, value: Any) -> None:
    st.session_state[key.value] = value


def set_item_safe(key: Keys, value: Any) -> None:
    """
    Assigns value to the given key if the key does not exist.
    :param key:
    :param value:
    :return:
    """
    if key.value in st.session_state:
        return

    set_item(key, value)


def del_item(key: Keys) -> None:
    if not contains(key):
        return

    del st.session_state[key.value]


def contains(key: Keys) -> bool:
    return key.value in st.session_state
