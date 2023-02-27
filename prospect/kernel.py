from abc import ABC, abstractmethod
import numpy as np 

def initialize_kernel(config_kernel):
    if config_kernel['type'] == 'montepython':
        kernel = MontePythonKernel(config_kernel)
    elif config_kernel['type'] == 'cobaya':
        raise NotImplementedError('')
    elif config_kernel['type'] == 'analytical':
        kernel = AnalyticalKernel(config_kernel)
    else:
        raise ValueError('You have specified a non-existing kernel type.')
    return kernel

class BaseKernel(ABC):
    def __init__(self, config_kernel):
        self.initialize(config_kernel)
        pass

    @abstractmethod
    def initialize(self, config_kernel):
        pass

    @abstractmethod
    def loglkl(self, position):
        pass

class AnalyticalKernel(BaseKernel):
    # Analytical likelihoods for development, testing and benchmarking purposes
    def initialize(self, config_kernel):
        self.config_kernel = config_kernel
        if config_kernel['function'] == 'gaussian':
            self.loglkl = self.Gaussian
        else:
            raise ValueError('No analytical kernel with the desired function type.')
        self.dimension = config_kernel['dimension']
    
    def Gaussian(self, position):
        loglkl = -0.5*np.dot(position - self.config_kernel['mean'], np.matmul(np.linalg.inv(self.config_kernel['covmat']), position - self.config_kernel['mean']))
        return loglkl
    
    def loglkl(self, position):
        return self.loglkl(position)

class MontePythonKernel(BaseKernel):
    def initialize(self, config_kernel):
        from montepython_public.montepython.data import Data
        import montepython_public.montepython.mcmc as mcmc 
        # Create a MontePython command_line from the contents in config 

    def loglkl(self, position):
        # wrap around mcmc.chain()
        pass


class CobayaKernel(BaseKernel):
    def initialize(self, config_kernel):
        raise NotImplementedError('cobaya kernel not yet implemented!')

    def loglkl(self, position):
        pass