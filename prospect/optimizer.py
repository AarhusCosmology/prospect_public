from abc import ABC, abstractmethod
from tasks import OptimizeTask, TaskStatus

class BaseOptimizer(ABC):
    def __init__(self, kernel, config_optimizer):
        self.kernel = kernel
        self.config = config_optimizer
    
    @abstractmethod
    def optimize():
        pass

class SimulatedAnnealing(BaseOptimizer):
    def __init__(self, kernel, config_optimizer):
        super().__init__(kernel, config_optimizer)
        # Create the kernel parameter file 
    
    def optimize(optimize_task): 
        optimize_task.status = TaskStatus.IN_PROGESS
        # Do optimization

        optimize_task.status = TaskStatus.FINISHED
        return optimize_task