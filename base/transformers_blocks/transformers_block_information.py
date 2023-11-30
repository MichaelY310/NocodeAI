from transformers import *

transformers_models_map = {
    "CLIPModel": [CLIPModel, CLIPConfig],
    "CLIPTextModel": [CLIPTextModel, CLIPTextConfig],
}

transformers_models_pretrained_choices_map = {
    "CLIPModel": ["", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
    "CLIPTextModel": ["", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
}

transformers_tokenizers_map = {
    "CLIPTokenizer": CLIPTokenizer,

}

transformers_tokenizers_pretrained_choices_map = {
    "CLIPTokenizer": ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
}