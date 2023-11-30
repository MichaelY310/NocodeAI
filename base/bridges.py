from prototype_classes import Bridge


class Bridges_Loop:
    def __init__(self, repeats=1):
        self.repeats = repeats
        self.bridges = []

    def insert_bridge(self, bridge, idx):
        self.bridges.insert(idx, bridge)

    def append_bridge(self, bridge):
        self.bridges.append(bridge)

    def forward(self):
        for repeat in range(self.repeats):
            for bridge in self.bridges:
                bridge.forward()


class Forward_Bridge(Bridge):
    def __init__(self, prev_block, next_block, paramName=""):
        super().__init__(prev_block, next_block)
        self.paramName = paramName

    def set_paramName(self, paramName):
        self.paramName = paramName

    def forward(self):
        s = ""
        if self.prev_block.playground.print_flow:
            s = str(self.prev_block.in_data)
        true_args = self.prev_block.combine_forward_parameters()
        self.prev_block.out_data = self.prev_block.forward(**true_args)
        if self.paramName == "":
            self.next_block.in_data += self.prev_block.out_data
        else:
            self.next_block.in_data.append(["*||*", self.paramName, self.prev_block.out_data])
        if self.prev_block.playground.print_flow:
            print(f"{self.prev_block.custom_name} is forwarded, === data is:  {s}  === output is: {self.prev_block.out_data}")
        if len(self.next_block.next_blocks) == 0:
            self.next_block.flow_and_pass()


class Accumulate_Forward_Bridge(Forward_Bridge):
    def __init__(self, prev_block, next_block, paramName=""):
        super().__init__(prev_block, next_block, paramName)

    def forward(self):
        super().forward()


class Clear_Forward_Bridge(Forward_Bridge):
    def __init__(self, prev_block, next_block, paramName=""):
        super().__init__(prev_block, next_block, paramName)

    def forward(self):
        self.next_block.in_data = []
        super().forward()


class Sleep_Bridge(Bridge):
    def __init__(self, prev_block, next_block):
        super().__init__(prev_block, next_block)

    def forward(self):
        return


# Call the method of the previous core block. Doesn't collect any output. Used for making some setting for certain core_modules. For example: Scheduler.set_timesteps(50)
class Call_Method_Bridge(Bridge):
    def __init__(self, prev_block, next_block, func_name=""):
        super().__init__(prev_block, next_block)
        self.func_name = func_name

    def set_func_name(self, func_name):
        self.func_name = func_name

    def forward(self):
        self.prev_block.get_core_module_attribute(self.func_name)(*self.next_block.forward())