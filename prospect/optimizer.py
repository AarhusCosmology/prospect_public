from abc import ABC, abstractmethod

def initialize_optimizer(config_optimizer, kernel):
    if config_optimizer['type'] == 'simulated_annealing':
        optimizer = SimulatedAnnealing(config_optimizer, kernel)
    else:
        raise ValueError('You have specified a non-existing optimizer type.')
    return optimizer

class BaseOptimizer(ABC):
    def __init__(self, config_optimizer, kernel):
        self.kernel = kernel
        self.config = config_optimizer
    
    @abstractmethod
    def optimize():
        pass

class SimulatedAnnealing(BaseOptimizer):
    def __init__(self, config_optimizer, kernel):
        super().__init__(config_optimizer, kernel)
        # Create the kernel parameter file 
    
    def optimize(self): 
        # Do optimization
        pass