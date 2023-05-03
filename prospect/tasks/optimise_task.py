from prospect.tasks.base_task import BaseTask
from prospect.kernels.initialise import initialise_kernel
from prospect.optimiser import initialise_optimiser

class OptimiseTask(BaseTask):
    priority = 25.0

    def __init__(self, config: dict, param_sample: float):
        super().__init__()
        self.config = config
        self.profile_parameter_value = param_sample

    def run(self):
        self.kernel = initialise_kernel(self.config['kernel'])
        self.optimiser = initialise_optimiser(self.config['optimizer'], self.kernel)
        self.optimiser.optimise()
        self.data = 0