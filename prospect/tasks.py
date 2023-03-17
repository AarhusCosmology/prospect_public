from abc import ABC, abstractmethod
from typing import Any, Type
from numpy import ndarray
from prospect.kernel import initialize_kernel
from prospect.mcmc import initialize_mcmc
from prospect.optimizer import initialize_optimizer

class BaseTask(ABC):
    idx_count = 0
    priority: float 
    required_task_ids: list[int]
    data: Any # Result from self.run() method is stored here

    def __init__(self, required_task_ids=[]):
        self.id = BaseTask.idx_count
        for req in required_task_ids:
            if req >= self.id:
                raise ValueError(f'Cannot require non-existent task of ID {req}.')
            elif req < 0:
                raise ValueError('Required task ID is negative.')
        self.required_task_ids = required_task_ids
        BaseTask.idx_count += 1

    @abstractmethod
    def run(self) -> None:
        pass

    def emit_tasks(self):
        # Any required information must be stored in self.data during call to run()
        return []

    def run_return_self(self, *args):
        self.run(*args)
        return self

    def __lt__(self, other):
        # Largest numerical value of priority is greatest
        return self.priority > other.priority

    @property
    def type(self) -> str:
        return self.__class__.__name__


class OptimizeTask(BaseTask):
    priority = 25.0

    def __init__(self, config: dict, param_sample: float):
        super().__init__()
        self.config = config
        self.profile_parameter_value = param_sample

    def run(self):
        self.kernel = initialize_kernel(self.config['kernel'])
        self.optimizer = initialize_optimizer(self.config['optimizer'], self.kernel)
        self.optimizer.optimize()
        self.data = 0


class MCMCTask(BaseTask):
    priority = 25.0

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self) -> ndarray:
        print("Running MCMCTask...")
        self.kernel = initialize_kernel(self.config['kernel'])
        self.mcmc = initialize_mcmc(self.config['mcmc'], self.kernel)
        self.mcmc.chain(self.config['mcmc']['N_steps'])
        self.data = self.mcmc.positions

class AnalyseMCMCTask(BaseTask):
    priority = 75.0

    def __init__(self, config: dict, required_task_ids: list[int]):
        super().__init__(required_task_ids)
        self.config = config

    def run(self, *chain_list: list[ndarray]):
        print("Running AnalyseMCMCTask... (to be implemented)")
        self.data = 0

    def emit_tasks(self) -> list[Type[BaseTask]]:
        return [MCMCTask(self.config) for idx in range(3)]

    def write_chains(self) -> None:
        pass

def initialize_tasks(config: dict) -> list[Type[BaseTask]]:
    if config['jobtype'] == 'profile':
        new_tasks = initial_profile_tasks(config)
    elif config['jobtype'] == 'mcmc':
        new_tasks = initial_mcmc_tasks(config)
    else:
        raise ValueError('No tasks to start at initialization. Check your input!')
    return new_tasks

def initial_profile_tasks(config: dict) -> list[Type[BaseTask]]:
    samples = []
    if config['profile']['dimension'] == '1d':
        if config['profile']['sampling_strategy']['type'] == 'manual':
            from prospect.sampling import ManualSampling
            samples = ManualSampling(config['profile']['sampling_strategy'])
        else:
            raise NotImplementedError('Only manual sampling is implemented currently.')
    else:
        raise NotImplementedError('Only 1d profiles are implemented currently.')
    task_list = []
    for sample in samples:
        task_list.append(OptimizeTask(config, sample))
    return task_list

def initial_mcmc_tasks(config: dict) -> list[Type[BaseTask]]:
    # When implementing MCMC, should make a more general function that returns 
    # a batch of MCMCTasks and their accompanying AnalyseMCMCTask
    task_list = []
    for idx_chain in range(config['mcmc']['N_chains']):
        task_list.append(MCMCTask(config))
    task_list.append(AnalyseMCMCTask(config, [task.id for task in task_list]))
    return task_list
