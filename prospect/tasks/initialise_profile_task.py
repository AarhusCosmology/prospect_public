from typing import Type
from prospect.input import Configuration
from prospect.tasks.base_task import BaseTask
from prospect.tasks.initialise_optimiser_task import InitialiseOptimiserTask

class InitialiseProfileTask(BaseTask):
    """
        Task which finds a sampling of the profile and emits optimizations at each point in the profile.

    """
    priority = 90.0

    def run(self, _):
        pass
    
    def emit_tasks(self) -> list[Type[BaseTask]]:
        task_list = []
        for idx_rep in range(self.config.profile.repetitions):
            for param_val in self.config.profile.values:
                task_list.append(InitialiseOptimiserTask(self.config, param_val, idx_rep))
        return task_list

def initialise_profile_tasks(config: Configuration) -> list[Type[BaseTask]]:
    return [InitialiseProfileTask(config)]