from .model import ModelBase, CodeLlama, GPT4, GPT35, StarChat, GPTDavinci
from .utils import Generator

from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .go_generate import GoGenerator


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        return RsGenerator()
    elif lang == "go" or lang == "golang":
        return GoGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo-0613":
        return GPT35()
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs) # add a specific version information
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
