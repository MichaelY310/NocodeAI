import inspect

import torch
import torch.nn as nn
from prototype_classes import Layer_Block, Playground

# {'in_features': 0, 'out_features': 0, 'bias': True, 'device': 'cuda', 'dtype': torch.float32}
class Linear_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("Linear", playground)
        self.min_dim = 1
        self.max_dim = -1
        self.inflow_attribute = {-1: [[f"any, ..., any, {self.parameters['in_features']}", "tensor"]]}
        self.outflow_attribute = {-1: [[f"any, ..., any, {self.parameters['out_features']}", "tensor"]]}


class Conv2d_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("Conv2d", playground)
        self.min_dim = 3
        self.max_dim = 4
        self.inflow_attribute = {3: [[f"{self.parameters['in_channels']}, any, any", "tensor"]], 4: [[f"any, {self.parameters['in_channels']}, any, any", "tensor"]]}
        self.outflow_attribute = {3: [[f"{self.parameters['out_channels']}, any, any", "tensor"]], 4: [[f"any, {self.parameters['out_channels']}, any, any", "tensor"]]}


class BatchNorm2d_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("BatchNorm2d", playground)
        self.min_dim = 3
        self.max_dim = 4


class ReLU_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("ReLU", playground)
        self.min_dim = 1
        self.max_dim = -1


class MaxPool2d_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("MaxPool2d", playground)
        self.min_dim = 1
        self.max_dim = -1


class Upsample_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("Upsample", playground)
        self.min_dim = 1
        self.max_dim = -1


class ConvTranspose2d_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("ConvTranspose2d", playground)
        self.min_dim = 1
        self.max_dim = -1


if __name__ == "__main__":
    playground = Playground("test")
    a = Conv2d_Block(playground)
    print(a.forward_parameters)
