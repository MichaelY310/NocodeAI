import inspect
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import transforms
from os import path
from prototype_classes import Data_Block, Image_Transformation, Block, CustomDataset_for_Image_Dataset_Block_for_single_folder_with_RE
from PIL import Image


class String_Block(Data_Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"string": str})
        self.min_dim = -1
        self.max_dim = -1
        self.inflow_attribute = -1
        self.outflow_attribute = {-1: [[-1, "str"]]}


class Tensor_Block(Data_Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"tensor": torch.Tensor})
        self.min_dim = -1
        self.max_dim = -1
        self.inflow_attribute = -1
        self.outflow_attribute = {-1: [[f"any, ..., any", "tensor"]]}


class Image_Block(Data_Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"img_path": str})
        self.min_dim = -1
        self.max_dim = -1
        self.inflow_attribute = -1
        self.outflow_attribute = {-1: [[f"3 or 1, any, any", "tensor"]]}
        self.image_transformations = [Image_Transformation("ToTensor", self.playground)]

    def get_img_path(self):
        return self.parameters["img_path"]

    def forward(self, *args):
        file_path = self.get_img_path()
        if not path.exists(self.parameters["img_path"]):
            print(f"{file_path} doesn't exist")
            return
        try:
            img = Image.open(file_path)
        except:
            print(f"{file_path} is a not picture, or there's something wrong with the picture")
            return
        image_transform = transforms.Compose(self.image_transformations)
        return image_transform(img)

    def add_transformation(self, idx, image_transformation_name):
        image_transformation = Image_Transformation(image_transformation_name, self.playground)
        self.image_transformations.insert(idx, image_transformation)


# Data processors:

class PIL_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = -1
        self.max_dim = -1
        self.inflow_attribute = -1
        self.outflow_attribute = {-1: [[-1, "PIL"]]}

    def get_img_path(self):
        return self.parameters["img_path"]

    def forward(self, *args, **kwargs):
        file_path = self.in_data[0]
        if not path.exists(file_path):
            print(f"{file_path} doesn't exist")
            return
        try:
            img = Image.open(file_path)
        except:
            print(f"{file_path} is a not picture, or there's something wrong with the picture")
            return
        return [img]


class Image_Transformations_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = -1
        self.max_dim = -1
        self.inflow_attribute = {-1: [[-1, "PIL"]]}
        self.outflow_attribute = {-1: [[f"3 or 1, any, any", "tensor"]]}
        self.image_transformations = [Image_Transformation("ToTensor", self.playground)]

    def forward(self, *args, **kwargs):
        image_transform = transforms.Compose(self.image_transformations)
        return [image_transform(*self.in_data)]

    def add_transformation(self, idx, image_transformation_name):
        image_transformation = Image_Transformation(image_transformation_name, self.playground)
        self.image_transformations.insert(idx, image_transformation)


# Dataloader and Datasets

class Dataloader_Block(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.min_dim = -1
        self.max_dim = -1
        self.inflow_attribute = -1
        self.outflow_attribute = {-1: [[-1, "dataloader"]]}
        self.set_parameters_types({'batch_size': int,
                                   'shuffle': bool,
                                   'sampler': Sampler,
                                   'batch_sampler': Sampler,
                                   'num_workers': int,
                                   'collate_fn': callable,
                                   'pin_memory': bool,
                                   'drop_last': bool,
                                   'timeout': float,
                                   'worker_init_fn': callable})
        self.set_parameters_default({'batch_size': 1,
                                     'shuffle': False,
                                     'sampler': None,
                                     'batch_sampler': None,
                                     'num_workers': 0,
                                     'collate_fn': None,
                                     'pin_memory': False,
                                     'drop_last': False,
                                     'timeout': 0,
                                     'worker_init_fn': None})
        self.use_default_parameters()
        self.dataloader = None
        self.dataloader_iterator = None

    def forward(self, *args, **kwargs):
        if self.dataloader is None:
            params = dict(self.parameters)
            params["dataset"] = self.in_data[0]
            self.dataloader = torch.utils.data.DataLoader(**params)
            self.dataloader_iterator = iter(self.dataloader)
        return next(self.dataloader_iterator)


# image dataloader that split image data in ONE SINGLE FOLDER according to the RE expression of the filenames
class Image_Dataset_Block_for_single_folder_with_RE(Block):
    def __init__(self, playground):
        super().__init__(playground)
        self.set_parameters_types({"root dir": str, "re expressions": list})
        self.min_dim = -1
        self.max_dim = -1
        self.inflow_attribute = -1
        self.outflow_attribute = {-1: [[-1, "dataset"]]}
        self.image_transformations = [Image_Transformation("ToTensor", self.playground)]
        self.dataset = None

    def get_root_dir(self):
        return self.parameters["root dir"]

    def get_re_expressions(self):
        return self.parameters["re expressions"]

    def add_transformation(self, idx, image_transformation_name):
        image_transformation = Image_Transformation(image_transformation_name, self.playground)
        self.image_transformations.insert(idx, image_transformation)

    def forward(self, *args, **kwargs):
        if self.dataset is None:
            transformation = transforms.Compose(self.image_transformations)
            root_dir = self.get_root_dir()
            re_expressions = self.get_re_expressions()
            self.dataset = CustomDataset_for_Image_Dataset_Block_for_single_folder_with_RE(root_dir, re_expressions, transformation)
        return [self.dataset]




