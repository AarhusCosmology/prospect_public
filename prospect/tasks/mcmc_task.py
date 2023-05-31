import os
from typing import Type
from prospect.kernels.initialisation import initialise_kernel
from prospect.mcmc import initialise_mcmc
from prospect.tasks.base_task import BaseTask
from prospect.analysis import get_gelman_rubin, getdist_gelman_rubin, analyse_mcmc
from prospect.input import Configuration
from prospect.io import unpack_mcmc

import time

class MCMCTask(BaseTask):
    priority = 25.0

    def __init__(self, config: Configuration, **mcmc_args):
        super().__init__()
        self.config = config
        self.mcmc_args = mcmc_args

    def run(self, _) -> None:
        tic = time.perf_counter()
        kernel = initialise_kernel(self.config.kernel, self.config.io.dir, self.id)
        self.mcmc = initialise_mcmc(self.config.mcmc, kernel, **self.mcmc_args)
        if self.config.mcmc.steps_per_iteration is not None:
            self.mcmc.run_steps(self.config.mcmc.steps_per_iteration)
        elif self.config.mcmc.minutes_per_iteration is not None:
            self.mcmc.run_minutes(self.config.mcmc.minutes_per_iteration)
        self.mcmc.finalize()
        toc = time.perf_counter()
        print(f"Finished MCMCTask of id {self.id} in {toc - tic:.2} seconds") # write to log instead 

class AnalyseMCMCTask(BaseTask):
    priority = 75.0

    def __init__(self, config: Configuration, required_task_ids: list[int]):
        super().__init__(required_task_ids)
        self.config = config
        self.dir = os.path.join(config.io.dir, config.run.jobtype)

    def run(self, mcmc_tasks: list[MCMCTask]): 
        tic = time.perf_counter()
        chains = [task.mcmc.chain for task in mcmc_tasks]
        kernel = initialise_kernel(self.config.kernel, self.config.io.dir, self.id)

        if self.config.mcmc.unpack_at_dump:
            unpack_mcmc(kernel.param, self.dir, self.config.io.jobname, *chains)
            if self.config.mcmc.analyse_automatically:
                analyse_mcmc(self.dir, self.config.io.jobname)

        # Discrepancy wrt. the GetDist Gelman-Rubin; mine is stricter. Use GetDist's!
        if self.config.mcmc.unpack_at_dump:
            # Problem with this: It only works when unpacking the MCMC!
            self.conv = getdist_gelman_rubin(f"{self.dir}/{self.config.io.jobname}")
        else:
            raise ValueError("Currently cannot compute R-1 without unpacking MCMCs. Please set 'unpack_at_dump' to True.")
            self.conv = get_gelman_rubin(chains)

        self.mcmc_tasks = mcmc_tasks
        self.initial_positions = [chain.last_position for chain in chains]
        toc = time.perf_counter()
        print(f"Finished AnalyseMCMCTask of id {self.id} in {toc - tic:.3} seconds") # write to log instead

    def emit_tasks(self) -> list[Type[BaseTask]]:
        if self.conv < self.config.mcmc.convergence_Rm1:
            print(f"Max. R-1 of {self.conv:.3} is below requirement {self.config.mcmc.convergence_Rm1}, finishing MCMC.")
            return [] # possibly emit some new finalising task 
        else:
            print(f"Max. R-1 of {self.conv:.3} is above requirement {self.config.mcmc.convergence_Rm1}, continuing MCMC.")
            return continue_mcmc_tasks(self.config, self.mcmc_tasks)

def initialise_mcmc_tasks(config: Configuration, mcmc_id: int) -> list[Type[BaseTask]]:
    task_list = []
    mcmc_args = {'mcmc_id': mcmc_id}
    for idx_chain in range(config.mcmc.N_chains):
            task_list.append(MCMCTask(config, **mcmc_args))
    task_list.append(AnalyseMCMCTask(config, [task.id for task in task_list]))
    return task_list

def continue_mcmc_tasks(config: Configuration, mcmc_tasks: list[MCMCTask]) -> list[Type[BaseTask]]:
    task_list = []
    for mcmc_task in mcmc_tasks:
        mcmc_args = {
            'mcmc_id': mcmc_task.mcmc_args['mcmc_id'],
            'chain': mcmc_task.mcmc.chain
        }
        task_list.append(MCMCTask(config, **mcmc_args))
    task_list.append(AnalyseMCMCTask(config, [task.id for task in task_list]))
    return task_list
