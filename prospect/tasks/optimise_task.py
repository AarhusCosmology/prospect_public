from typing import Type
import numpy as np 
from prospect.input import Configuration
from prospect.tasks.base_task import BaseTask
from prospect.kernels.initialisation import initialise_kernel
from prospect.optimiser import initialise_optimiser

class OptimiseTask(BaseTask):
    priority = 25.0

    def __init__(self, config: Configuration, optimise_task_settings: dict):
        super().__init__(config)
        self.optimise_settings = optimise_task_settings

    def run(self, _):
        kernel = initialise_kernel(self.config.kernel, self.config.io.dir, self.id)
        if 'fixed_param_val' in self.optimise_settings:
            kernel.set_fixed_parameters({self.config.profile.parameter: self.optimise_settings['fixed_param_val']})
        self.optimiser = initialise_optimiser(self.config, kernel, self.optimise_settings)
        self.optimiser.optimise()
        self.optimiser.set_bestfit()
        self.optimiser.finalize()

        print(f"OptimiseTask id={self.id}, iteration {self.optimise_settings['iteration_number']} at {self.config.profile.parameter}={self.optimise_settings['fixed_param_val']:.4e} finished:")
        print(f"\tBestfit loglkl={self.optimiser.bestfit['loglkl']:.10} | Acceptance rate={self.optimiser.bestfit['acceptance_rate']:.4} | Num_evals={np.sum(self.optimiser.mcmc.chain.mults)} | Temp={self.optimise_settings['temperature']:.4e} | Step size={self.optimise_settings['step_size']:.4e}")

    def emit_tasks(self) -> list[Type[BaseTask]]:
        loglkl_improvement = self.optimise_settings['current_best_loglkl'] - self.optimiser.bestfit['loglkl']
        if loglkl_improvement < 0.:
            print(f"Best chi2 didn't improve since last iteration ({self.optimise_settings['current_best_loglkl']:.10}). Emitting next OptimiseTask.")
        elif self.optimise_settings['current_best_loglkl'] == np.inf:
            # First iteration, continue
            pass
        elif 2*loglkl_improvement > self.config.profile.chi2_tolerance:
            print(f"Best chi2 improved by {2*loglkl_improvement:.4e} since last iteration.\nEmitting new OptimiseTask since this is larger than the tolerance chi2_tolerance={self.config.profile.chi2_tolerance}.")
        else:
            print(f"Best chi2 improved by {2*loglkl_improvement:.4e} since last iteration.\nFinishing optimisation since this is less than the tolerance chi2_tolerance={self.config.profile.chi2_tolerance}.")
            return []
        new_settings = self.optimiser.get_next_iteration_settings()
        new_settings['iteration_number'] = self.optimise_settings['iteration_number'] + 1
        new_settings['repetition_number'] = self.optimise_settings['repetition_number']
        return [OptimiseTask(self.config, new_settings)]
