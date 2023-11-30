from prototype_classes import *


class No_Grad_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        self.playground.no_grad = True
        return self.in_data


class Calc_Grad_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        self.playground.no_grad = False
        return self.in_data


class Get_Global_Variable_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"global variable": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_global_variable(self):
        return self.parameters["global variable"]

    def forward(self, *args, **kwargs):
        return [self.playground.get_global_variable(self.get_global_variable())]


class Increase_Global_Variable_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"global variable": str, "increment": int})
        self.min_dim = 1
        self.max_dim = -1

    def get_global_variable(self):
        return self.parameters["global variable"]

    def get_increment(self):
        return self.parameters["increment"]

    def forward(self, *args, **kwargs):
        self.playground.global_variables[self.get_global_variable()] += self.get_increment()
        return None


class Call_Method_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        func = self.in_data[0]
        args = self.in_data[1:]
        parameters_types, parameters_default, parameters = generate_parameters3_for_function(self, func)

        actual_args = dict(parameters)
        ks = list(parameters.keys())
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
        print(actual_args)
        return [func(**actual_args)]


class Len_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        res = []
        for i in range(0, len(self.in_data)):
            res.append(len(self.in_data[i]))
        return res


class Sum_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        res = self.in_data[0]
        for i in range(1, len(self.in_data)):
            res += self.in_data[i]
        return [res]


class Multiply_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        res = self.in_data[0]
        for i in range(1, len(self.in_data)):
            res *= self.in_data[i]
        return [res]


class Floor_Divide_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_forward_parameters_types({"dividend": any, "divisor": any})
        initialize_parameters_for_forward(self)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, dividend, divisor):
        return [dividend // divisor]


class Negative_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        res = []
        for i in range(0, len(self.in_data)):
            res.append(-self.in_data[i])
        return res


class Idx_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_forward_parameters_types({"target": any, "idx": typing.Union[int, str]})
        initialize_parameters_for_forward(self)
        self.min_dim = 1
        self.max_dim = -1

    def get_idx(self):
        return self.parameters["idx"]

    def forward(self, target, idx):
        return [target[idx]]


class Pack_List_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        return [self.in_data]


# get the attribute of the core module in the previous block
class Attribute_Block_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"attribute name": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_attribute_name(self):
        return self.parameters["attribute name"]

    def forward(self, *args, **kwargs):
        return [self.prev_blocks[0].get_core_module_attribute(self.get_attribute_name())]


# get the attribute of the data output of the previous block
class Attribute_Data_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"attribute name": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_attribute_name(self):
        return self.parameters["attribute name"]

    def forward(self, *args, **kwargs):
        return [getattr(self.in_data[0], self.get_attribute_name())]


# get the attribute of the core module in the previous block
class Tensor_Size_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        res = []
        for i in self.in_data:
            res.append(i.size())
        return res


class Random_Tensor_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"dtype": torch.dtype})
        initialize_parameters_for_class(self)
        self.set_forward_parameters_types({"shape": list})
        initialize_parameters_for_forward(self)
        self.min_dim = 1
        self.max_dim = -1

    def get_dtype(self):
        return self.parameters["dtype"]

    def forward(self, shape):
        t = torch.randn(shape, dtype=self.get_dtype(), device=self.default_device)
        return [t]


class Assign_ParamName_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"parameter name": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_ParamName(self):
        return self.parameters["parameter name"]

    def forward(self, *args, **kwargs):
        return [["*||*", self.get_ParamName(), self.in_data]]


class Tensor_Cat_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1

    def forward(self, *args, **kwargs):
        return [torch.cat(self.in_data)]

# class Tensor_Cat_Block(Block):
#     def __init__(self, playground):
#         super().__init__(playground)
#         self.min_dim = 1
#         self.max_dim = -1
#         self.core_function = torch.cat
#         initialize_parameters_for_forward_with_core_function(self)
#
#     def forward(self, *args, **kwargs):
#         return [self.core_function(**kwargs)]


class List_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"list": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_list(self):
        return self.parameters["list"]

    def forward(self, *args, **kwargs):
        e = f"""global l
l = {self.get_list()}"""
        exec(e)
        return [l]


class Bool_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"bool": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_bool(self):
        return self.parameters["bool"]

    def forward(self, *args, **kwargs):
        return [self.get_bool() == "True"]


class Float_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"float": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_float(self):
        return self.parameters["float"]

    def forward(self, *args, **kwargs):
        return [float(self.get_float())]


class Int_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"int": str})
        self.min_dim = 1
        self.max_dim = -1

    def get_bool(self):
        return self.parameters["int"]

    def forward(self, *args, **kwargs):
        return [int(self.get_bool())]


class ToTensor_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1
        self.set_forward_parameters_types({"data": any, "dtype": torch.dtype})
        initialize_parameters_for_forward(self)
        self.set_forward_parameters_default({"data": any, "dtype": torch.float})

    def get_data(self):
        return self.forward_parameters["data"]

    def get_dtype(self):
        return self.forward_parameters["dtype"]

    def forward(self, *args, **kwargs):
        return [torch.tensor(device=self.default_device, **kwargs)]


class Chunk_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1
        self.set_forward_parameters_types({"input": torch.Tensor, "chunks": int, "dim": int})
        initialize_parameters_for_forward(self)

    def get_input(self):
        return self.forward_parameters["input"]

    def get_chunks(self):
        return self.forward_parameters["chunks"]

    def get_dim(self):
        return self.forward_parameters["dim"]

    def forward(self, *args, **kwargs):
        return [torch.chunk(**kwargs)]


class Save_PIL_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = 1
        self.max_dim = -1
        self.set_forward_parameters_types({"path": str})
        initialize_parameters_for_forward(self)

    def get_path(self):
        return self.forward_parameters["path"]

    def forward(self, *args, **kwargs):
        self.in_data[0].save(self.get_path())
        return self.in_data


if __name__ == "__main__":
    playground = Playground("test")
    a = Chunk_Block(playground)
    # a.set_forward_parameters({"shape": [1,2,3]})
    # print(a.flow_and_pass())