from torchvision.transforms.transforms import *
from torch.nn import *

class_names = [
    "Compose",
    "ToTensor",
    "PILToTensor",
    "ConvertImageDtype",
    "ToPILImage",
    "Normalize",
    "Resize",
    "CenterCrop",
    "Pad",
    "Lambda",
    "RandomApply",
    "RandomChoice",
    "RandomOrder",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomResizedCrop",
    "FiveCrop",
    "TenCrop",
    "LinearTransformation",
    "ColorJitter",
    "RandomRotation",
    "RandomAffine",
    "Grayscale",
    "RandomGrayscale",
    "RandomPerspective",
    "RandomErasing",
    "GaussianBlur",
    "InterpolationMode",
    "RandomInvert",
    "RandomPosterize",
    "RandomSolarize",
    "RandomAdjustSharpness",
    "RandomAutocontrast",
    "RandomEqualize",
    "ElasticTransform",
]

for clas in class_names:
    c = globals()[clas]
    print(f"'{clas}': tranforms.{c.__name__},")