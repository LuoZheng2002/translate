from enum import Enum, auto


from typing import NamedTuple, Union

class Language(Enum):
    CHINESE = auto()
    HINDI = auto()

class TranslateMode(Enum):
    DATASET_FULLY_TRANSLATED_PROMPT_DEFAULT = auto()
    DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE = auto()
    DATASET_PARTIALLY_TRANSLATED = auto()


class Translated(NamedTuple):
    language: Language
    translate_mode: TranslateMode    

class NotTranslated(NamedTuple):
    pass

TranslateInfo = Union[Translated, NotTranslated]



class AddNoiseMode(Enum):
    NO_NOISE = auto()
    ADD_NOISE = auto()

class Config:
    def __init__(self, translate_info: TranslateInfo, add_noise_mode: AddNoiseMode):
        self.translate_info = translate_info
        self.add_noise_mode = add_noise_mode

configs: list[Config] = [
    Config(translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
]

requires_inference = True
requires_evaluation = True
requires_score = True