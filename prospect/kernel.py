from abc import ABC, abstractmethod

class BaseKernel(ABC):
    def __init__(self, config_kernel):
        self.initialize(config_kernel)
        pass

    @abstractmethod
    def initialize(config_kernel):
        pass

class MontePythonKernel(BaseKernel):
    def initialize(self, config_kernel):
        pass