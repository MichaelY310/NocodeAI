import inspect
import os
import typing
from block_information import *
import random


def get_filenames_from_folder(root_dir):
    filenames = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


def get_parameterName_defaultValue_type_list(f):
    parameterNames = []
    defaultValues = []
    types = []
    sig = inspect.signature(f)
    for param in sig.parameters.values():
        parameterNames.append(param.name)
        defaultValues.append(param.default)
        types.append(param.annotation)

    return parameterNames, defaultValues, types


def extract_parameter_types_from_class(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.__init__)
    parameters_types = {}
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            parameters_types["device"] = "dtype choice"
        # dtype
        elif pName == "dtype":
            parameters_types["dtype"] = "dtype choice"
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][-1]
            parameters_types[pName] = union_type
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                parameters_types[parameterNames[i]] = type(pValue)
            else:
                parameters_types[parameterNames[i]] = pType
        # common types
        else:
            parameters_types[parameterNames[i]] = pType
    return parameters_types


def generate_parameter_values_from_types(Block, parameters_types):
    module_parameters = {}
    parameterNames = list(parameters_types.keys())
    for i in range(0, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameters_types[pName]
        # device
        if pName == "device":
            module_parameters["device"] = Block.default_device
        # dtype
        elif pName == "dtype":
            module_parameters["dtype"] = dtype_choice["torch.float"]
        # common types
        else:
            module_parameters[pName] = default_value_handlers_map[pType]
    return module_parameters


def generate_parameter_values_from_class(Block, Class):
    parameterNames, parameterDefaultValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.__init__)
    module_parameters = {}
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterDefaultValues[i]
        # device
        if pName == "device":
            module_parameters["device"] = Block.default_device
        # dtype
        elif pName == "dtype":
            module_parameters["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][-1]
            module_parameters[pName] = default_value_handlers_map[union_type]
            # if there is already a default but doesn't fit into the type, turn all the values of that type into that default value
            if str(pValue) != "<class 'inspect._empty'>":
                module_parameters[pName] = (pValue,) * len(module_parameters[pName])
        # common types
        else:
            # if there is a default value, then just use it. If not, use the default_value_handlers_map to generate a default value
            if str(pValue) != "<class 'inspect._empty'>":
                module_parameters[pName] = pValue
            else:
                module_parameters[pName] = default_value_handlers_map[pType]
    return module_parameters


# set parameter types before use this function
# set parameters_default and parameters
# it doesn't set the in_data
def initialize_parameters_for_class(Class):
    Class.parameters_default = {}
    for pName, pType in Class.parameters_types.items():
        # device
        if pName == "device":
            Class.parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.parameters_default["dtype"] = dtype_choice["torch.float"]
        # common types
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type} is not in default_value_handlers_map")
            Class.parameters_default[pName] = default_value_handlers_map[union_type]
        else:
            Class.parameters_default[pName] = default_value_handlers_map[pType]
            if hasattr(Class.parameters_default[pName], "to"):
                Class.parameters_default[pName] = Class.parameters_default[pName].to(Class.default_device)
    Class.parameters = Class.parameters_default


def initialize_parameters_for_forward(Class):
    Class.parameters_default = {}
    for pName, pType in Class.forward_parameters_types.items():
        # device
        if pName == "device":
            Class.forward_parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.forward_parameters_default["dtype"] = dtype_choice["torch.float"]
        # common types
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.forward_parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type} is not in default_value_handlers_map")
            Class.forward_parameters_default[pName] = default_value_handlers_map[union_type]
        else:
            Class.forward_parameters_default[pName] = default_value_handlers_map[pType]
            if hasattr(Class.forward_parameters_default[pName], "to"):
                Class.forward_parameters_default[pName] = Class.forward_parameters_default[pName].to(Class.default_device)
    Class.forward_parameters = Class.forward_parameters_default


def initialize_parameters_for_Layer_Block(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.core_module_class.__init__)
    Class.parameters_types = {}
    Class.parameters_default = {}
    # skip the parameter "self"
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.parameters_types["device"] = "device choice"
            Class.parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.parameters_types["dtype"] = "dtype choice"
            Class.parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.parameters_default[pName] = pValue

        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.parameters_types[parameterNames[i]] = type(pValue)
                Class.parameters_default[pName] = pValue
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.parameters = Class.parameters_default


def initialize_parameters_for_Image_Transformation(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.core_module_class.__init__)
    Class.parameters_types = {}
    Class.parameters_default = {}
    # skip the parameter "self"
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.parameters_types["device"] = "device choice"
            Class.parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.parameters_types["dtype"] = "dtype choice"
            Class.parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.parameters_types[parameterNames[i]] = type(pValue)
                Class.parameters_default[pName] = pValue
            elif pName == "size":
                Class.parameters_types[parameterNames[i]] = typing.Tuple[int, int]
                Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.parameters = Class.parameters_default


def initialize_parameters_for_Loss_Function(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.core_module_class.__init__)
    Class.parameters_types = {}
    Class.parameters_default = {}
    # skip the parameter "self"
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.parameters_types["device"] = "device choice"
            Class.parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.parameters_types["dtype"] = "dtype choice"
            Class.parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.parameters_types[parameterNames[i]] = type(pValue)
                Class.parameters_default[pName] = pValue
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.parameters = Class.parameters_default


def initialize_parameters_for_Optimizer_Block(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.core_module_class.__init__)
    Class.parameters_types = {}
    Class.parameters_default = {}
    # skip the parameter "self"
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.parameters_types["device"] = "device choice"
            Class.parameters_default["device"] = Class.default_device
        elif pName == "lr":
            Class.parameters_types["lr"] = float
            Class.parameters_default["lr"] = float(0.001)
        elif pName == "params":
            Class.parameters_types["target network"] = str
            Class.parameters_default["target network"] = ""
        # dtype
        elif pName == "dtype":
            Class.parameters_types["dtype"] = "dtype choice"
            Class.parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.parameters_types[parameterNames[i]] = type(pValue)
                Class.parameters_default[pName] = pValue
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.parameters = Class.parameters_default


def initialize_parameters_for_Diffusers_Scheduler(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.core_module_class.__init__)
    Class.parameters_types = {}
    Class.parameters_default = {}
    # skip the parameter "self"
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.parameters_types["device"] = "device choice"
            Class.parameters_default["device"] = Class.default_device
        elif pName == "lr":
            Class.parameters_types["lr"] = float
            Class.parameters_default["lr"] = float(0.001)
        elif pName == "params":
            Class.parameters_types["target network"] = str
            Class.parameters_default["target network"] = ""
        # dtype
        elif pName == "dtype":
            Class.parameters_types["dtype"] = "dtype choice"
            Class.parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.parameters_types[parameterNames[i]] = type(pValue)
                Class.parameters_default[pName] = pValue
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.parameters_types[parameterNames[i]] = pType
            Class.parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.parameters = Class.parameters_default


def initialize_parameters_for_Transformers_Config(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.config_class.__init__)
    Class.config_parameters_types = {}
    Class.config_parameters_default = {}
    # skip the parameter "self"
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.config_parameters_types["device"] = "device choice"
            Class.config_parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.config_parameters_types["dtype"] = "dtype choice"
            Class.config_parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.config_parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.config_parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.config_parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.config_parameters_types[parameterNames[i]] = type(pValue)
                Class.config_parameters_default[pName] = pValue
            elif pName == "kwargs":
                continue
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.config_parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.config_parameters_types[parameterNames[i]] = pType
            Class.config_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.config_parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.config_parameters_types[parameterNames[i]] = pType
            Class.config_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.config_parameters = Class.config_parameters_default


def initialize_parameters_for_forward_with_core(Class):
    core_forward_Class = Class.core_module_class
    try:
        parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(core_forward_Class.forward)
    except:
        parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(core_forward_Class.__call__)
    # skip the parameter "self"
    for i in range(1, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.forward_parameters_types["device"] = "device choice"
            Class.forward_parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.forward_parameters_types["dtype"] = "dtype choice"
            Class.forward_parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.forward_parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.forward_parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.forward_parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.forward_parameters_types[parameterNames[i]] = type(pValue)
                Class.forward_parameters_default[pName] = pValue
            elif pName == "kwargs":
                continue
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.forward_parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.forward_parameters_types[parameterNames[i]] = pType
            Class.forward_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.forward_parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.forward_parameters_types[parameterNames[i]] = pType
            Class.forward_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.forward_parameters = Class.forward_parameters_default


def initialize_parameters_for_forward_with_core_function(Class):
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(Class.core_function)
    # skip the parameter "self"
    for i in range(0, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            Class.forward_parameters_types["device"] = "device choice"
            Class.forward_parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            Class.forward_parameters_types["dtype"] = "dtype choice"
            Class.forward_parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            Class.forward_parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            Class.forward_parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                Class.forward_parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                Class.forward_parameters_types[parameterNames[i]] = type(pValue)
                Class.forward_parameters_default[pName] = pValue
            elif pName == "kwargs":
                continue
            # elif pName == "":
            #     Class.parameters_types[parameterNames[i]] = ##fill this part with type##
            #     Class.parameters_default[pName] = default_value_handlers_map[Class.parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                Class.forward_parameters_types[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            Class.forward_parameters_types[parameterNames[i]] = pType
            Class.forward_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                Class.forward_parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            Class.forward_parameters_types[parameterNames[i]] = pType
            Class.forward_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    Class.forward_parameters = Class.forward_parameters_default



def generate_parameters3_for_function(Class, function):
    res_parameters_types = {}
    res_parameters_default = {}
    res_parameters = {}
    parameterNames, parameterValues, parameterTypes = get_parameterName_defaultValue_type_list(function)
    # skip the parameter "self"
    for i in range(0, len(parameterNames)):
        pName = parameterNames[i]
        pType = parameterTypes[i]
        pValue = parameterValues[i]
        # device
        if pName == "device":
            res_parameters_types["device"] = "device choice"
            res_parameters_default["device"] = Class.default_device
        # dtype
        elif pName == "dtype":
            res_parameters_types["dtype"] = "dtype choice"
            res_parameters_default["dtype"] = dtype_choice["torch.float"]
        # Union Type
        elif type(pType) == typing._UnionGenericAlias:
            union_type = [arg for arg in pType.__args__][0]
            res_parameters_types[pName] = pType
            if union_type not in default_value_handlers_map:
                print(f"{pName}\n{pType}\n{union_type}\n{pValue} is not in default_value_handlers_map")
            res_parameters_default[pName] = default_value_handlers_map[union_type]
            if str(pValue) != "<class 'inspect._empty'>":
                res_parameters_default[pName] = pValue
        elif pType == inspect._empty:
            if pValue != inspect._empty:
                res_parameters_types[parameterNames[i]] = type(pValue)
                res_parameters_default[pName] = pValue
            elif pName == "kwargs":
                continue
            # elif pName == "":
            #     res_parameters_types[parameterNames[i]] = ##fill this part with type##
            #     res_parameters_default[pName] = default_value_handlers_map[res_parameters_types[parameterNames[i]]]
            else:
                print(pName, " has no type or default value, please make a special case for it")
                res_parameters_default[parameterNames[i]] = pType
        # common types
        elif pType in default_value_handlers_map:
            res_parameters_types[parameterNames[i]] = pType
            res_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
            if pValue != inspect._empty:
                res_parameters_default[pName] = pValue
        else:
            print("unexpected datatype occurred:   ", pType, "   in the parameter:  ", pName)
            res_parameters_types[parameterNames[i]] = pType
            res_parameters_default[parameterNames[i]] = default_value_handlers_map[pType]
    res_parameters = res_parameters_default
    return res_parameters_types, res_parameters_default, res_parameters


def generate_random_string(length):
    res = ""
    for i in range(length):
        res += chr(random.randint(65, 90))
    return res


def create_forward_with_name_and_code(class_name, code):
    e = f"""def forward(*args):
"""
    code = code.split("\n")
    for i in code:
        e += "    "
        e += i
        e += "\n"
    with open(f"my_classes\\{class_name}.py", "w", encoding='utf-8') as f:
        f.write(e)


def delete_forward_with_name(class_name):
    os.remove(f"my_classes\\{class_name}.py")


def get_forward_from_string(string):
    e = f"""from my_classes.{string} import forward
global return_class
return_class = forward"""
    exec(e)
    return return_class


if __name__ == "__main__":
    code = """
    print("hello world")
    print("操你妈逼")
    """
    create_forward_with_name_and_code("Class_Number_One", code)
    c = get_forward_from_string("Class_Number_One")
    c()