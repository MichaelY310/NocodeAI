from transformers_blocks.transformers_prototype_classes import Transformer_Tokenizer_Block


class CLIPTokenizer_Block(Transformer_Tokenizer_Block):
    def __init__(self, playground):
        super().__init__("CLIPTokenizer", playground)

