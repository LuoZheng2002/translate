from enum import Enum, auto


from typing import NamedTuple, Union

class Model(Enum):
    GPT_4O_MINI = auto()
    CLAUDE_SONNET = auto()  #
    CLAUDE_HAIKU = auto()  #

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
    def __init__(self, model: Model, translate_info: TranslateInfo, add_noise_mode: AddNoiseMode):
        self.model = model
        self.translate_info = translate_info
        self.add_noise_mode = add_noise_mode

configs: list[Config] = [
    Config(model=Model.GPT_4O_MINI, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    Config(model=Model.GPT_4O_MINI, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_FULLY_TRANSLATED_PROMPT_DEFAULT), add_noise_mode=AddNoiseMode.NO_NOISE),
    Config(model=Model.GPT_4O_MINI, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.ADD_NOISE),
    Config(model=Model.GPT_4O_MINI, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_FULLY_TRANSLATED_PROMPT_DEFAULT), add_noise_mode=AddNoiseMode.ADD_NOISE),
    Config(model=Model.CLAUDE_SONNET, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    Config(model=Model.CLAUDE_SONNET, translate_info=Translated(language=Language.HINDI, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    Config(model=Model.CLAUDE_SONNET, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.ADD_NOISE),
    Config(model=Model.CLAUDE_SONNET, translate_info=Translated(language=Language.HINDI, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.ADD_NOISE),
]

requires_inference = True
requires_evaluation = True
requires_score = True