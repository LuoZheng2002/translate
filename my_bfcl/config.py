from enum import Enum, auto
from dataclasses import dataclass

from typing import NamedTuple, Union

class ApiModel(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    CLAUDE_HAIKU = "claude-haiku-4-5"

class LocalModel(Enum):
    GRANITE_3_1_8B_INSTRUCT = "ibm-granite/granite-3.1-8b-instruct"

@dataclass
class LocalModelStruct:
    model: LocalModel
    generator: any = None  # Placeholder for the actual generator object

Model = Union[ApiModel, LocalModelStruct]

class Language(Enum):
    CHINESE = auto()
    HINDI = auto()

class TranslateOption(Enum):
    DATASET_FULLY_TRANSLATED = auto()
    DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE = auto()
    DATASET_PARTIALLY_TRANSLATED = auto()


@dataclass(frozen=True)
class Translated:
    language: Language
    option: TranslateOption

@dataclass(frozen=True)
class NotTranslated:
    pass

TranslateMode = Union[Translated, NotTranslated]



class AddNoiseMode(Enum):
    NO_NOISE = auto()
    SYNONYM = auto()
    PARAPHRASE = auto()

# class Config:
#     def __init__(self, model: Model, translate_info: TranslateMode, add_noise_mode: AddNoiseMode):
#         self.model = model
#         self.translate_info = translate_info
#         self.add_noise_mode = add_noise_mode

@dataclass(frozen=True)
class Config:
    model: Model
    translate_info: TranslateMode
    add_noise_mode: AddNoiseMode

configs: list[Config] = [
    # Config(model=Model.GPT_4O_MINI, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=Model.GPT_4O_MINI, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=Model.GPT_4O_MINI, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.ADD_NOISE),
    # Config(model=Model.GPT_4O_MINI, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.ADD_NOISE),
    # Config(model=Model.CLAUDE_SONNET, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=Model.CLAUDE_SONNET, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=Model.CLAUDE_SONNET, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.ADD_NOISE),
    # Config(model=Model.CLAUDE_SONNET, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.ADD_NOISE),
    # Config(model=ApiModelStruct(model=Model.CLAUDE_HAIKU), translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModelStruct(model=Model.CLAUDE_HAIKU), translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModelStruct(model=Model.CLAUDE_HAIKU), translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.ADD_NOISE),
    # Config(model=ApiModelStruct(model=Model.CLAUDE_HAIKU), translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.ADD_NOISE),
    Config(model=LocalModelStruct(model=LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, translate_info=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, translate_info=Translated(language=Language.CHINESE, translate_mode=TranslateMode.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.ADD_NOISE),

]

requires_inference = True
requires_evaluation = True
requires_score = True