from abc import ABC, abstractmethod
from kernel import initialize_kernel
from mcmc import initialize_mcmc
from optimizer import initialize_optimizer

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
TaskStatus = enum('NOT_STARTED', 'IN_PROGESS', 'FINISHED')
TaskTags = enum('READY', 'DONE', 'EXIT', 'START')

class BaseTask(ABC):
    idx_count = 0
    def __init__(self, required_task_ids=[]):
        self.status = TaskStatus.NOT_STARTED
        self.id = BaseTask.idx_count
        BaseTask.idx_count += 1
        self.assigned_worker = None
        self.required_task_ids = required_task_ids

    @abstractmethod
    def run():
        pass

    def load(self, required_tasks):
        if len(required_tasks) == len(self.required_task_ids):
            for id in self.required_task_ids:
                assert(id in self.required_task_ids)
            for task in required_tasks:
                assert(task.status == TaskStatus.FINISHED)
        else:
            raise Exception('Task was not loaded with the required list of finished tasks')
        self.required_tasks = required_tasks
        
class OptimizeTask(BaseTask):
    def __init__(self, config, param_sample):
        super().__init__()
        self.config = config
        self.profile_parameter_value = param_sample

    def run(self):
        self.kernel = initialize_kernel(self.config['kernel'])
        self.optimizer = initialize_optimizer(self.config['optimizer'], self.kernel)
        self.optimizer.optimize()
        self.status = TaskStatus.FINISHED

class MCMCTask(BaseTask):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        self.kernel = initialize_kernel(self.config['kernel'])
        self.mcmc = initialize_mcmc(self.config['mcmc'], self.kernel)
        self.mcmc.chain(self.config['mcmc']['N_steps'])
        self.status = TaskStatus.FINISHED

# from analysis import mcmc_diagnostics
class AnalyseMCMCTask(BaseTask):
    def __init__(self, config, required_task_ids):
        super().__init__(required_task_ids)
        self.config = config

    def run(self):
        print("ANALYSIS TASK BEING RUN")
        # from io import write_mcmc
        # write_mcmc(self.required_tasks, OPTIONS)
        # print(self.required_tasks)
        self.status = TaskStatus.FINISHED
    
    def write_chains(self):
        pass 

class AcquisitionTask(BaseTask):
    pass

class AnalyseTask(BaseTask):
    pass