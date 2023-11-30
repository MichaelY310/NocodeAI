import torch
import numpy
import torchvision.transforms as transforms

class To_Tensor_Block:
    def __init__(self):
        self.id = 3
        self.category = "img preprocessor"
        self.category_id = 0
        self.info = ""
        self.input = "any"
        self.output = "tensor"

    def forward(self, data):
        transform_method = transforms.ToTensor()
        return transform_method