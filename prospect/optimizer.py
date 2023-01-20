from abc import ABC, abstractmethod

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
    
    def optimize(self): 
        # Do optimization
        return 0.0