import inspect
from PIL import Image
import torch
import torch.nn as nn
from layers import Linear_Block, Conv2d_Block
from loss_functions import MSELoss_Block
from utils import get_parameterName_defaultValue_type_list, get_filenames_from_folder
from prototype_classes import Playground, Optimizer_Block, Integrated_Network_Block
from torchvision import transforms
from data import *
from loss_functions import MSELoss_Block
from layers import *
from data import *
from calcs import *
import torch
from transformers_blocks.transformers_tokenizers import *
from transformers_blocks.transformers_models import *
from diffusers_blocks.diffusers_schedulers import *
from diffusers_blocks.diffusers_models import *
from bridges import *

playground = Playground("playground01")
print(playground.default_device)

# for i in get_parameterName_defaultValue_type_list(torch.nn.Linear):
#     print(i)

x1 = torch.tensor([[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]], dtype=torch.float, device="cuda")
x2 = torch.tensor([[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]], dtype=torch.float, device="cuda")
img2 = [
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],

    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
]
x2 = torch.tensor(img2, dtype=torch.float, device="cuda")


# img = Image.open("scarlet_house.jpg")
# totensor = Image_Transformation("ToTensor", playground)
# transform = transforms.Compose([
#     totensor,
#     transforms.ConvertImageDtype(dtype=torch.float32),
# ])
#
# print()
#
# img = transform(img)
# print(img.shape)


# image_block = Image_Block(playground)
# image_block.set_parameters({"img_path": "scarlet_house.jpg"})
# image_block.add_transformation(0, "RandomCrop")
# image_block.image_transformations[0].set_parameters({'size': (4, 4), 'padding': None, 'pad_if_needed': False, 'fill': 0, 'padding_mode': 'constant'})
# print(image_block.flow_out())

# network = Integrated_Network_Block(playground)
# network.set_custom_name("network block")
# print(playground.block_map)
# optimizer_block = Optimizer_Block("SGD", playground)
# optimizer_block.set_parameters({'target network': 'network block', 'lr': 0.001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False})
# optimizer_block.set_custom_name("optimizer block")
# optimizer_block.print_parameters_info()

# dataset = CustomDataset_for_Image_Dataset_Block_for_single_folder_with_RE("D:\\datasets\\train", ["cat.*", "dog.*"])
# print(dataset)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, timeout=3.333)
# for i in dataloader:
#     print(i)
#     break
# print(dataloader)

from diffusers import *
# vae1 = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae')
# unet1 = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet')

# vae = AutoencoderKL_Block(playground)
# unet = UNet2DConditionModel_Block(playground)

class a:
    def __init__(self):
        self.val = 0

    def set_val(self, val):
        self.val = val

p = a()
s = getattr(p, "set_val")
print(s)
s(10)
print(p.val)