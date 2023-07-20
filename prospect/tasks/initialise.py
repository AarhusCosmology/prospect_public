from typing import Type
from prospect.tasks.base_task import BaseTask
from prospect.tasks.mcmc_task import initialise_mcmc_tasks
from prospect.tasks.initialise_profile_task import initialise_profile_tasks

def initialise_tasks(config) -> list[Type[BaseTask]]:
    if config.run.jobtype == 'mcmc':
        new_tasks = initialise_mcmc_tasks(config, mcmc_id=0)
    elif config.run.jobtype == 'profile':
        new_tasks = initialise_profile_tasks(config)
    else:
        raise ValueError(f'No tasks to start at initialization with jobtype {config["jobtype"]}.')
    return new_tasks