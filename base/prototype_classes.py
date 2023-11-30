import glob
import os
import re

import torch.nn as nn
import typing

from PIL import Image
from torch.utils.data import Dataset

from block_information import *
from utils import *


class Playground:
    def __init__(self, playground_name, mode="train"):
        self.playground_name = playground_name
        self.block_map = {}
        self.block_count = 0
        self.mode = mode
        self.no_grad = False
        self.default_device = "gpu" if torch.cuda.is_available() else "cpu"
        self.code_block_count = 0
        self.print_flow = True
        self.tensor_shape_only = False
        self.flow_order = []
        self.global_variables = {}

    def start(self):
        for i in self.flow_order:
            if self.no_grad:
                with torch.no_grad:
                    i.forward()
            else:
                i.forward()

    def add_bridge(self, bridge):
        self.flow_order.append(bridge)

    def add_loop(self, loop):
        self.flow_order.append(loop)

    def delete_global_variable(self, string):
        del self.global_variables[string]

    def get_global_variable(self, string):
        return self.global_variables[string]

    def add_global_variable(self, string):
        self.global_variables[string] = None

    def set_global_variable(self, string, val):
        self.global_variables[string] = val

    def add_independent_block(self, block, name=""):
        self.block_map[self.block_count] = block
        block.playground_id = self.block_count
        self.block_count += 1


# 一个 Block 拥有两种 parameters. 一个是初始化模块核心的parameters， 一个是调用模块核心forward函数所需要的forward_parameters
# 传入数据本质上是填充 forward 所需的 parameters. 而用户也可以在传入数据之前设置 forward 的 parameters.
class Block(nn.Module):
    def __init__(self, playground):
        super().__init__()
        self.playground = playground
        self.last_prev_block = None
        self.last_next_block = None
        self.prev_blocks = []
        self.next_blocks = []
        self.in_data = []
        self.out_data = None
        self.custom_name = ""
        self.inflow_attribute = None
        self.outflow_attribute = None
        # 用户选择：cpu or gpu
        self.default_device = ""
        self.set_device(playground.default_device)
        self.head_block_only = False
        self.tail_block_only = False
        self.not_head_block = False
        self.not_tail_block = False
        self.parameters = {}
        self.parameters_default = {}
        self.parameters_types = {}
        # there is no parameter by default
        self.set_parameters_types({})
        self.use_default_parameters()

        self.forward_parameters_types = {}
        self.forward_parameters_default = {}
        self.forward_parameters = {}

    def set_parameters_types(self, parameters_types):
        self.parameters_types = parameters_types
        # overwrite this line for classes with a core algorithm
        initialize_parameters_for_class(self)

    def set_parameters_default(self, parameters_default):
        self.parameters_default = parameters_default
        self.use_default_parameters()

    # the default parameters are automatically set into the device
    def use_default_parameters(self):
        self.parameters = self.parameters_default

    # user can use it fill in the parameter values
    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_forward_parameters_types(self, forward_parameters_types):
        self.forward_parameters_types = forward_parameters_types
        initialize_parameters_for_forward(self)

    def set_forward_parameters_default(self, forward_parameters_default):
        self.forward_parameters_default = forward_parameters_default

    def use_default_forward_parameters(self):
        self.forward_parameters = self.forward_parameters_default

    def set_forward_parameters(self, forward_parameters):
        self.forward_parameters = forward_parameters

    def get_core_module_attribute(self, attribute_name: str):
        if not hasattr(self, "core_module"):
            print(f"{self.custom_name} has no core_module")
            return None
        if not hasattr(self.core_module, attribute_name):
            print(f"there is not attribute {attribute_name} in {self.core_module} in {self.custom_name}")
            return None
        return getattr(self.core_module, attribute_name)

    def print_parameters_info(self):
        print("parameters: \n", self.parameters)
        print("parameter types: \n", self.parameters_types)
        print("default parameters: \n", self.parameters_default)
        print("number of the parameters: \n", len(self.parameters.keys()))

    # 把自己加在模块的后面
    # add this block after another block
    def add_self_after_block(self, block):
        if self.head_block_only:
            print(f"{self.custom_name} must be a head block")
            return
        block.next_blocks.append(self)
        self.prev_blocks.append(block)

    def add_block_after_self(self, block):
        if self.tail_block_only:
            print(f"{self.custom_name} must be a tail block")
        block.prev_blocks.append(self)
        self.next_blocks.append(block)

    def combine_forward_parameters(self):
        args = self.in_data
        actual_args = dict(self.forward_parameters)
        ks = list(self.forward_parameters.keys())
        index = 0
        already_set = set()
        if len(ks) < len(args):
            return {}
        for i in range(len(args)):
            current_arg = args[i]
            if type(current_arg) == list and len(current_arg) == 3 and current_arg[0] == "*||*":
                param_name = current_arg[1]
                value = current_arg[2]
                actual_args[param_name] = value[0]
                already_set.add(param_name)
            else:
                actual_args[ks[index]] = current_arg
                already_set.add(ks[index])
            while index < len(ks) and ks[index] in already_set:
                index += 1
        return actual_args

    # to be overwritten
    def forward(self, *args, **kwargs):
        return self.combine_forward_parameters()

    # flow_out first gathers all the out flow of the previous blocks, then passes them as parameters to the function forward, which is the main algorithm part, and finally returns an output
    def flow_and_pass(self):
        s = ""
        if self.playground.print_flow:
            s = str(self.in_data)
        true_args = self.combine_forward_parameters()
        self.out_data = self.forward(**true_args)
        for block in self.next_blocks:
            block.in_data += self.out_data
        if self.playground.print_flow:
            print(f"{self.custom_name} is forwarded, === data is:  {s}  === output is: {self.out_data}")
        return self.out_data

    def set_custom_name(self, custom_name):
        if self.custom_name in self.playground.block_map:
            del self.playground.block_map[self.custom_name]
        self.custom_name = custom_name
        self.playground.block_map[self.custom_name] = self

    def set_device(self, device):
        if device == "gpu":
            self.default_device = "cuda"
        else:
            self.default_device = device

    def __call__(self, *args):
        return self.forward(*args)


class Bridge:
    def __init__(self, prev_block, next_block):  # mode: accumulate or clear. clear means delete all the in_data in the next_block
        self.prev_block = prev_block
        self.next_block = next_block
        next_block.add_self_after_block(prev_block)

    def forward(self):
        s = ""
        if self.prev_block.playground.print_flow:
            s = str(self.prev_block.in_data)
        true_args = self.prev_block.combine_forward_parameters()
        self.prev_block.out_data = self.prev_block.forward(**true_args)
        self.next_block.in_data += self.prev_block.out_data
        if self.prev_block.playground.print_flow:
            print(f"{self.prev_block.custom_name} is forwarded, === data is:  {s}  === output is: {self.prev_block.out_data}")
        if len(self.next_block.next_blocks) == 0:
            self.next_block.flow_and_pass()


class Backward_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.train = True

    def backward(self, *args):
        for i in args:
            i.backward()

    def forward(self, *args, **kwargs):
        if self.playground.mode == "train":
            for i in self.in_data:
                self.backward(i)
        return self.in_data


class Optimizer_Block(Block):
    def __init__(self, optimizer_name, playground):
        super().__init__(playground)
        self.core_module_name = optimizer_name
        self.core_module_class = optimizer_map[self.core_module_name]
        initialize_parameters_for_Optimizer_Block(self)
        parameters_ = dict(self.parameters)
        parameters_["params"] = None
        del parameters_["target network"]
        self.core_module = None

    def set_parameters(self, parameters):
        self.parameters = parameters
        parameters_ = dict(self.parameters)
        parameters_["params"] = self.playground.block_map[parameters_["target network"]].core_module.parameters()
        del parameters_["target network"]
        self.core_module = self.core_module_class(**parameters_)
        self.core_module.zero_grad()

    def forward(self, *args, **kwargs):
        self.core_module.step()
        self.core_module.zero_grad()
        return args

    def get_core_module_attribute(self, attribute_name: str):
        return getattr(self.core_module, attribute_name)


class Merge_Block(Block):
    def forward(self, *args, **kwargs):
        return self.in_data


# must be a head block
class Data_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        # self.inflow_attribute = "forbidden"
        self.head_block_only = True
        self.set_parameters_types({"data": None})
        initialize_parameters_for_class(self)
        self.in_data.append(self.parameters["data"])

    def set_parameters_types(self, parameters_types):
        self.parameters_types = parameters_types
        # overwrite this line for classes with a core algorithm
        initialize_parameters_for_class(self)
        self.set_parameters(self.parameters)

    # set the in_data as soon as the user set the parameters. Also automatically put the data in the device
    def set_parameters(self, parameters):
        self.parameters = parameters
        self.in_data = []
        for param in list(self.parameters.values()):
            if hasattr(param, "to"):
                self.in_data.append(param.to(self.default_device))
            else:
                self.in_data.append(param)

    def forward(self, *args, **kwargs):
        return self.in_data


class Layer_Block(Block):
    def __init__(self, layer_name, playground):
        super().__init__(playground)
        self.core_module_name = layer_name
        self.core_module_class = layer_map[self.core_module_name]
        initialize_parameters_for_Layer_Block(self)
        initialize_parameters_for_forward_with_core(self)
        self.core_module = self.core_module_class(**self.parameters)

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.core_module = self.core_module_class(**self.parameters)

    def forward(self, *args, **kwargs):
        return [self.core_module.forward(**kwargs)]


class Integrated_Network_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.sub_layer_blocks = []
        self.core_module = nn.ModuleList()

    def add_layer(self, idx, layer_block):
        self.sub_layer_blocks.insert(idx, layer_block)
        self.core_module.insert(idx, layer_block.core_module)

    def forward(self, *args, **kwargs):
        self.sub_layer_blocks[0].in_data = self.in_data
        for i in range(0, len(self.sub_layer_blocks)-1):
            self.sub_layer_blocks[i].add_block_after_self(self.sub_layer_blocks[i+1])
            self.sub_layer_blocks[i].flow_and_pass()
        return self.sub_layer_blocks[-1].flow_and_pass()


class Customized_Network_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.sub_layer_blocks = []
        self.core_module = nn.ModuleList()
        self.order = []

    def set_order(self, order):
        self.order = order

    def add_layer(self, idx, layer_block):
        self.sub_layer_blocks.insert(idx, layer_block)
        self.core_module.insert(idx, layer_block.core_module)

    def forward(self, *args, **kwargs):
        self.sub_layer_blocks[self.order[0]].in_data = self.in_data
        for i in self.order:
            self.sub_layer_blocks[i].flow_and_pass()
        return self.sub_layer_blocks[-1].flow_and_pass()


class Print_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)

    def forward(self, *args, **kwargs):
        print(f"========  {self.custom_name} start printing  ========")
        print(self.in_data)
        print(f"==============================================")
        return self.in_data


class ToGPU(Block):
    def __init__(self, playground):
        super().__init__(playground)

    def forward(self, *args, **kwargs):
        res = []
        for i in self.in_data:
            if type(i) == list and i[0] == "*||*":
                ii = list(i)
                for iii in range(len(ii[3])):
                    ii[3][iii] = ii[3][iii].to("cuda")
                res.append(ii)
            else:
                res.append(i.to("cuda"))
        return res


class Image_Transformation(Block):
    def __init__(self, image_transformation_name, playground):
        super().__init__(playground)
        self.playground = playground
        self.core_module_name = image_transformation_name
        self.core_module_class = image_transformations_map[self.core_module_name]
        initialize_parameters_for_Image_Transformation(self)
        initialize_parameters_for_forward_with_core(self)
        self.core_module = self.core_module_class(**self.parameters)

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.core_module = self.core_module_class(**self.parameters)

    def forward(self, *args, **kwargs):
        if len(args) != 0:
            return self.core_module(*args)
        else:
            return self.core_module(**kwargs)


class Loss_Function_Block(Block):
    def __init__(self, loss_function_name, playground):
        super().__init__(playground)
        self.core_module_name = loss_function_name
        self.playground = playground
        self.core_module_class = loss_function_map[self.core_module_name]
        initialize_parameters_for_Loss_Function(self)
        initialize_parameters_for_forward_with_core(self)
        self.core_module = self.core_module_class(**self.parameters)

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.core_module = self.core_module_class(**self.parameters)

    def forward(self, *args, **kwargs):
        print(kwargs)
        return [self.core_module.forward(**kwargs)]


class CustomDataset_for_Image_Dataset_Block_for_single_folder_with_RE(Dataset):
    def __init__(self, root_dir, re_expressions, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.re_expressions = re_expressions

        # Find all images in the root directory
        pictures = os.listdir(root_dir)

        # Assign labels to the images based on their names
        for picture in pictures:
            match = False
            for i in range(len(self.re_expressions)):
                if re.fullmatch(self.re_expressions[i], picture):
                    self.images.append(root_dir + '\\' + picture)
                    self.labels.append(i)
                    match = True
                    break
            if not match:
                dir = root_dir + '\\' + picture
                print(f"none of the re expressions matches {dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(self.__load_image(image))
        return image, label

    def __load_image(self, image_path):
        return Image.open(image_path)


class Code_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.playground.code_block_count += 1
        self.code = ""
        self.true_forward = None
        self.true_name = f"Code_Block_{self.playground.code_block_count}"

    def set_code(self, code):
        self.code = code

    def forward(self, *args, **kwargs):
        code = self.code
        code = code.split("\n")
        e = """global forward_
def forward_(self, *args, **kwargs):"""
        for c in code:
            e += f"\n\t{c}"
        exec(e)
        return [forward_(self, *self.in_data, **kwargs)]


# class Global_Variable_Block


if __name__ == "__main__":
    playground = Playground("test")
    a = Code_Block(playground)
    a.set_code("print('hello world')\nreturn args[0]")

    b = Code_Block(playground)
    b.set_code("print('fucking world')\nreturn args[1]")

    c = Code_Block(playground)
    c.set_code("return torch.tensor([0])")

    print(a.forward(114514, 100))
    print(b.forward(114514, 100))
    print(c.forward(114514, 100))