from abc import ABC, abstractmethod

def initialise_optimiser(config_optimiser, kernel):
    if config_optimiser['type'] == 'simulated_annealing':
        optimiser = SimulatedAnnealing(config_optimiser, kernel)
    else:
        raise ValueError('You have specified a non-existing optimizer type.')
    return optimiser

class BaseOptimiser(ABC):
    def __init__(self, config_optimiser, kernel):
        self.kernel = kernel
        self.config = config_optimiser
    
    @abstractmethod
    def optimize():
        pass

class SimulatedAnnealing(BaseOptimiser):
    def __init__(self, config_optimiser, kernel):
        super().__init__(config_optimiser, kernel)
        # Create the kernel parameter file 
    
    def optimise(self): 
        # Do optimisation
        pass