class Result:
    def __init__(
            self,
            index: int,
            text: str,
            model_prediction: str,
            real_state: str,
            is_user_right: bool,
            is_model_right: bool
    ):
        self.__index: int = index
        self.__text: str = text
        self.__model_prediction: str = model_prediction
        self.__real_answer: str = real_state
        self.__is_user_right: bool = is_user_right
        self.__is_model_right: bool = is_model_right

    @property
    def index(self) -> int: return self.__index

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