from diffusers_blocks.diffusers_prototype_classes import Diffuser_Model_Block


class AutoencoderKL_Block(Diffuser_Model_Block):
    def __init__(self, playground):
        super().__init__("AutoencoderKL", playground)
        self.set_parameters_default({"pretrained": self.pretrained_choices[0], "advanced_pretrained": {"subfolder": "vae"}})
        self.use_default_parameters()
        self.initialize_core()
        self.core_module = self.core_module.to(self.default_device)


class UNet2DConditionModel_Block(Diffuser_Model_Block):
    def __init__(self, playground):
        super().__init__("UNet2DConditionModel", playground)
        self.set_parameters_default({"pretrained": self.pretrained_choices[0], "advanced_pretrained": {"subfolder": "unet"}})
        self.use_default_parameters()
        self.initialize_core()
        self.core_module = self.core_module.to(self.default_device)
        print("set gpu", self.core_module_name, "to", self.default_device)
        print("current device is: ", self.core_module.device)