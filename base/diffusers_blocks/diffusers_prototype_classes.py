from prototype_classes import Block
from diffusers_blocks.diffusers_block_information import *
from utils import *


class Diffuser_Model_Block(Block):
    def __init__(self, model_name, playground):
        super().__init__(playground)
        self.core_module_name = model_name
        self.pretrained_choices = diffusers_models_pretrained_choices_map[self.core_module_name]
        self.set_parameters_types({"pretrained": str, "advanced_pretrained": dict})
        self.set_parameters_default({"pretrained": self.pretrained_choices[0], "advanced_pretrained": {}})
        self.use_default_parameters()
        self.core_module_class = diffusers_models_map[self.core_module_name]
        initialize_parameters_for_forward_with_core(self)
        self.core_module = None

    def initialize_core(self):
        self.core_module = self.core_module_class.from_pretrained(self.get_pretrained(), **self.get_advanced_pretrained())

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.core_module = self.core_module_class.from_pretrained(self.get_pretrained(), **self.get_advanced_pretrained())
        self.core_module = self.core_module.to(self.default_device)

    def get_pretrained(self):
        return self.parameters["pretrained"]

    def get_advanced_pretrained(self):
        return self.parameters["advanced_pretrained"]

    def forward(self, *args, **kwargs):
        return [self.core_module(**kwargs)]


class Diffuser_Scheduler_Block(Block):
    def __init__(self, scheduler_name, playground):
        super().__init__(playground)
        self.core_module_name = scheduler_name
        self.core_module_class = diffusers_schedulers_map[self.core_module_name]
        initialize_parameters_for_Diffusers_Scheduler(self)
        initialize_parameters_for_forward_with_core(self)
        self.core_module = self.core_module_class(**self.parameters)

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.core_module = self.core_module_class(**self.parameters)

    def forward(self, *args, **kwargs):
        print("diffuser scheduler has nothing to forward")
        return args