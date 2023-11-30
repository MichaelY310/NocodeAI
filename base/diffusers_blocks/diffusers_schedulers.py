from diffusers_blocks.diffusers_prototype_classes import Diffuser_Scheduler_Block


class LMSDiscreteScheduler_Block(Diffuser_Scheduler_Block):
    def __init__(self, playground):
        super().__init__("LMSDiscreteScheduler", playground)
