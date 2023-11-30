import torch
import torch.nn as nn
from prototype_classes import Layer_Block


class ReLU_Block(Layer_Block):
    def __init__(self, playground):
        super().__init__("relu", playground)
        self.min_dim = 1
        self.max_dim = -1
        self.inflow_attribute = {-1: [[f"any, ..., any", "tensor"]]}
        self.outflow_attribute = {-1: [[f"any, ..., any", "tensor"]]}


