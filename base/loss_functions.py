import torch
import torch.nn as nn
from prototype_classes import Loss_Function_Block


class MSELoss_Block(Loss_Function_Block):
    def __init__(self, playground):
        super().__init__("MSELoss", playground)
        self.min_dim = 1
        self.max_dim = -1
        self.inflow_attribute = {-1: [[f"any, ..., any", "tensor"], [f"any, ..., any", "tensor"]]}
        self.outflow_attribute = {"forbidden": [[f"any", "int"]]}
