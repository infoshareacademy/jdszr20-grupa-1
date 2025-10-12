from enum import Enum


class Result:
    def __init__(
            self,
            index_user: int,
            index_model: int,
            index_actual: int
    ):
        self.__index_user: int = index_user
        self.__index_model: int = index_model
        self.__index_actual: int = index_actual
        self.__is_user_right: bool = index_user == index_actual
        self.__is_model_right: bool = index_model == index_actual

    def __str__(self):
        return\
            (f"Actual: {self.__index_actual}\n"
             f"User: {self.__index_user}\n"
             f"Model: {self.__index_model}\n"
             f"Is user right: {self.__is_user_right}\n"
             f"Is model right: {self.__is_model_right}\n")

    @property
    def index_user(self) -> int: return self.__index_user

    @property
    def index_model(self) -> int: return self.__index_model

    @property
    def index_actual(self) -> int: return self.__index_actual

    @property
    def is_user_right(self) -> bool: return self.__is_user_right

    @property
    def is_model_right(self) -> bool: return self.__is_model_right


class Model(Enum):
    SVC = "SVC"
    XGBoost = "XGBoost"