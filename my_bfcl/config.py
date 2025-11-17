from enum import Enum, auto
from dataclasses import dataclass

from typing import NamedTuple, Union

class ApiModel(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    CLAUDE_HAIKU = "claude-haiku-4-5"
    DEEPSEEK_CHAT = "deepseek-chat"

class LocalModel(Enum):
    GRANITE_3_1_8B_INSTRUCT = "ibm-granite/granite-3.1-8b-instruct"

# @dataclass
# class LocalModelStruct:
#     model: LocalModel
#     generator: any = None  # Placeholder for the actual generator object

Model = Union[ApiModel, LocalModel]

class Language(Enum):
    CHINESE = auto()
    HINDI = auto()

class TranslateOption(Enum):
    DATASET_FULLY_TRANSLATED = auto()
    DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE = auto()
    DATASET_PARTIALLY_TRANSLATED = auto()
    DATASET_FULLY_TRANSLATED_POST_PROCESS = auto()


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
    translate_mode: TranslateMode
    add_noise_mode: AddNoiseMode

configs: list[Config] = [
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.PARAPHRASE),

    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.PARAPHRASE),
    
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),    
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    Config(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.PARAPHRASE),

    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.NO_NOISE),

]

requires_inference_raw = True
requires_inference_json = True
requires_post_processing = True # rephrase parameter values if the raw output has a similar meaning as the ground truth but is not an exact match
requires_evaluation = True
requires_score = True

evaluation_caching = True


