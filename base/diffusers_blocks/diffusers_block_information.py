from diffusers import *

diffusers_models_map = {
    "UNet2DConditionModel": UNet2DConditionModel,
    "AutoencoderKL": AutoencoderKL,
}

diffusers_models_pretrained_choices_map = {
    "UNet2DConditionModel": ['CompVis/stable-diffusion-v1-4'],
    "AutoencoderKL": ["CompVis/stable-diffusion-v1-4"],
}

diffusers_schedulers_map = {
    "LMSDiscreteScheduler": LMSDiscreteScheduler,
    "PNDMScheduler": PNDMScheduler
}

diffusers_pipelines = {
    "StableDiffusionPipeline": StableDiffusionPipeline,
}

diffusers_pipelines_pretrained_choices_map = {
    "StableDiffusion": "CompVis/stable-diffusion-v1-4",
}

StableDiffusionPipeline.from_pretrained
EulerDiscreteScheduler