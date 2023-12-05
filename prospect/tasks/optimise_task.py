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
        self.converged = False

    def run(self, _):
        kernel = initialise_kernel(self.config.kernel, self.config.io.dir, self.id)
        if 'fixed_param_val' in self.optimise_settings:
            kernel.set_fixed_parameters({self.config.profile.parameter: self.optimise_settings['fixed_param_val']})
        self.optimiser = initialise_optimiser(self.config, kernel, self.optimise_settings)
        self.optimiser.optimise()
        self.optimiser.set_bestfit()
        print(f"OptimiseTask id={self.id}, iteration {self.optimise_settings['iteration_number']} at {self.config.profile.parameter}={self.optimise_settings['fixed_param_val']:.4e} finished:")
        print(f"\tBestfit loglkl={self.optimiser.bestfit['loglkl']:.10} | Acceptance rate={self.optimiser.bestfit['acceptance_rate']:.4} | Num_evals={np.sum(self.optimiser.mcmc.chain.mults)} | Temp={self.optimise_settings['temperature']:.4e} | Step size={self.optimise_settings['step_size']:.4e}")

    def finalize(self):
        del self.optimiser.mcmc.kernel
        del self.optimiser.kernel

    def emit_tasks(self) -> list[Type[BaseTask]]:
        loglkl_improvement = self.optimise_settings['current_best_loglkl'] - self.optimiser.bestfit['loglkl']
        if self.config.profile.chi2_tolerance is not None and self.config.profile.chi2_tolerance != 0.0:
            if loglkl_improvement < 0.:
                print(f"Best chi2 didn't improve since last iteration ({self.optimise_settings['current_best_loglkl']:.10}). Emitting next OptimiseTask.")
            elif self.optimise_settings['current_best_loglkl'] == np.inf:
                # First iteration, continue
                pass
            # New convergence criterion: All loglkls appearing in chain must be < tolerance
            # Perhaps a better conv crit is not to compare bestfits but just the loglkl of the last points
            # or alternatively the Euclidean distance between last points (but this is difficult to normalize...)
            #elif np.any(2*np.array(self.optimiser.mcmc.chain.loglkls) > self.config.profile.chi2_tolerance):
            elif 2*loglkl_improvement > self.config.profile.chi2_tolerance:
                print(f"Best chi2 improved by {2*loglkl_improvement:.4e} since last iteration.\nEmitting new OptimiseTask since some loglkls deviated more than tolerance chi2_tolerance={self.config.profile.chi2_tolerance}.")
            elif self.optimiser.bestfit['accepted_steps'] > 1:
                print(f"Best chi2 improved by {2*loglkl_improvement:.4e} since last iteration.\nFinishing optimisation since all loglkls deviated less than tolerance chi2_tolerance={self.config.profile.chi2_tolerance}.")
                self.converged = True
                return []
        new_settings = self.optimiser.get_next_iteration_settings()
        new_settings['iteration_number'] = self.optimise_settings['iteration_number'] + 1
        new_settings['repetition_number'] = self.optimise_settings['repetition_number']
        return [OptimiseTask(self.config, new_settings)]
