from prototype_classes import Block
from transformers_blocks.transformers_block_information import *
from utils import *


class Transformer_Model_Block(Block):
    def __init__(self, model_name, playground):
        super().__init__(playground)
        self.core_module_name = model_name
        self.set_parameters_types({"pretrained": str, "advanced_pretrained": dict})
        self.config_class = transformers_models_map[self.core_module_name][1]
        self.core_module_class = transformers_models_map[self.core_module_name][0]
        self.config_parameters_types = {}
        self.config_parameters_default = {}
        self.config_parameters = {}
        initialize_parameters_for_Transformers_Config(self)
        initialize_parameters_for_forward_with_core(self)
        self.config = self.config_class(**self.config_parameters)
        if self.get_pretrained() == "":
            self.core_module = self.core_module_class(self.config)
        else:
            self.core_module = self.core_module_class.from_pretrained(self.get_pretrained(), **self.get_advanced_pretrained())

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.core_module = self.core_module_class.from_pretrained(self.get_pretrained(), **self.get_advanced_pretrained())
        self.core_module = self.core_module.to(self.default_device)

    def get_pretrained(self):
        return self.parameters["pretrained"]

    def get_advanced_pretrained(self):
        return self.parameters["advanced_pretrained"]

    def set_config_parameters(self, config_parameters):
        self.config_parameters = config_parameters
        self.config = self.config_class(**self.config_parameters)
        if self.get_pretrained() == "":
            self.core_module = self.core_module_class(self.config)
        else:
            print("the model is already pretrained. Configuration will not be used")

    def forward(self, *args, **kwargs):
        return [self.core_module(**kwargs)]


class Transformer_Tokenizer_Block(Block):
    def __init__(self, tokenizer_name, playground):
        super().__init__(playground)
        self.core_module_name = tokenizer_name
        self.pretrained_choices = transformers_tokenizers_pretrained_choices_map[self.core_module_name]
        self.set_parameters_types({"pretrained": str, "advanced_pretrained": dict})
        self.set_parameters_default({"pretrained": self.pretrained_choices[0], "advanced_pretrained": {}})
        self.use_default_parameters()
        self.core_module_class = transformers_tokenizers_map[self.core_module_name]
        self.core_module = self.core_module_class.from_pretrained(self.get_pretrained(), **self.get_advanced_pretrained())
        initialize_parameters_for_forward_with_core(self)

    def get_pretrained(self):
        return self.parameters["pretrained"]

    def get_advanced_pretrained(self):
        return self.parameters["advanced_pretrained"]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.core_module = self.core_module_class.from_pretrained(self.get_pretrained(), **self.get_advanced_pretrained())

    def forward(self, *args, **kwargs):
        return [self.core_module(**kwargs)]
