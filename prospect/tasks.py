from abc import ABC, abstractmethod

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
#TaskTypes = enum('OPTIMIZE', 'ACQUISITION')
TaskStatus = enum('NOT_STARTED', 'IN_PROGESS', 'FINISHED')
TaskTags = enum('READY', 'DONE', 'EXIT', 'START')

class BaseTask(ABC):
    idx_count = 0
    def __init__(self):
        self.status = TaskStatus.NOT_STARTED
        self.id = BaseTask.idx_count
        BaseTask.idx_count += 1
        self.assigned_worker = None

    @abstractmethod
    def run():
        pass
        

class OptimizeTask(BaseTask):
    def __init__(self, config, param_sample):
        super().__init__()
        self.config = config
        self.profile_parameter_value = param_sample

    def run(self):
        if self.config['kernel']['type'] == 'montepython':
            from kernel import MontePythonKernel
            self.kernel = MontePythonKernel(self.config['kernel'])
        elif self.config['kernel']['type'] == 'cobaya':
            raise NotImplementedError('')
        else:
            raise ValueError('You have specified a non-existing kernel type.')

        if self.config['optimizer']['type'] == 'simulated_annealing':
            from optimizer import SimulatedAnnealing
            self.optimizer = SimulatedAnnealing(self.kernel, self.config['optimizer'])
        else:
            raise ValueError('You have specified a non-existing optimizer type.')

        self.status = TaskStatus.FINISHED

class AcquisitionTask(BaseTask):
    pass

class AnalyseTask(BaseTask):
    pass