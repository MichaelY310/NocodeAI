from transformers_blocks.transformers_prototype_classes import Transformer_Model_Block


class CLIPModel_Block(Transformer_Model_Block):
    def __init__(self, playground):
        super().__init__("CLIPModel", playground)
        self.core_module = self.core_module.to(self.default_device)


class CLIPTextModel_Block(Transformer_Model_Block):
    def __init__(self, playground):
        super().__init__("CLIPTextModel", playground)
        self.core_module = self.core_module.to(self.default_device)
