from abc import ABC, abstractmethod
import numpy as np 
from prospect.mcmc import MetropolisHastings, Chain
from prospect.mcmc import Arguments as MCMCArguments

def initialise_optimiser(config, kernel, optimise_settings):
    if config.profile.optimiser == 'simulated annealing':
        optimiser = SimulatedAnnealing(config, kernel, optimise_settings)
    else:
        raise ValueError('You have specified a non-existing optimizer type.')
    return optimiser

class BaseOptimiser(ABC):
    def __init__(self, config, kernel, optimise_settings):
        self.kernel = kernel
        self.config = config
        self.settings = optimise_settings
        if 'initial_position' in optimise_settings and any(optimise_settings['initial_position']):
            self.initial_position = optimise_settings['initial_position']
        else:
            raise ValueError('No point to start the optimisation from.')
        if 'covmat' in optimise_settings:
            self.covmat = optimise_settings['covmat']
        else:
            raise ValueError('No covmat to start the optimisation from.')

    @abstractmethod
    def optimise():
        pass

    @abstractmethod
    def set_bestfit():
        pass

    @abstractmethod
    def get_next_iteration_settings():
        pass

class SimulatedAnnealing(BaseOptimiser):
    def __init__(self, config, kernel, optimise_settings):
        super().__init__(config, kernel, optimise_settings)

        config_mcmc = MCMCArguments(
            'Metropolis Hastings',
            1, # N_chains
            config.profile.steps_per_iteration,
            config.profile.minutes_per_iteration,
            0.0, # Target R-1; here we use different convergence criterion
            False, # unpack_at_dump
            self.settings['temperature'],
            self.settings['step_size'],
            False, # analyse_automatically
            self.covmat
        )
        mcmc_args = {
            'chain': Chain(
                            [1], 
                            [self.kernel.loglkl(self.initial_position)],
                            self.initial_position
                        )
        }
        self.mcmc = MetropolisHastings(config_mcmc, self.kernel, **mcmc_args)

    def optimise(self):
        print(f"Running a simulated annealing iteration of parameter {self.config.profile.parameter}={self.settings['fixed_param_val']} with temperature {self.settings['temperature']}")
        if self.config.profile.steps_per_iteration is not None:
            self.mcmc.run_steps(self.config.profile.steps_per_iteration)
        elif self.config.profile.minutes_per_iteration is not None:
            self.mcmc.run_minutes(self.config.profile.minutes_per_iteration)
    
    def set_bestfit(self):
        bestfit_idx = np.argmin(self.mcmc.chain.loglkls)
        self.bestfit = {
            'loglkl': self.mcmc.chain.loglkls[bestfit_idx],
            'acceptance_rate': len(self.mcmc.chain.loglkls)/np.sum(self.mcmc.chain.mults)
        }
        for param_name, param_vec in self.mcmc.chain.positions.items():
            self.bestfit[param_name] = param_vec[bestfit_idx]
        #print(f"Optimiser set bestfit loglkl={self.bestfit['loglkl']}. Chain length is {len(self.mcmc.chain.loglkls)}.")
    
    def finalize(self):
        self.mcmc.finalize()
        del self.kernel
    
    def get_next_iteration_settings(self):
        if self.config.profile.temperature_schedule == 'exponential':
            new_temp = self.settings['temperature']*self.settings['temperature_change']
        else:
            raise ValueError('Invalid temperature schedule.')
        if self.config.profile.step_size_schedule == 'exponential':
            new_step_size = self.settings['step_size']*self.settings['step_size_change']
        else:
            raise ValueError('Invalid temperature schedule.')
        
        optimise_settings = {
            'current_best_loglkl': self.bestfit['loglkl'],
            'fixed_param_val': self.settings['fixed_param_val'],
            'initial_position': self.mcmc.chain.last_position,
            'covmat': self.covmat,
            'temperature': new_temp,
            'temperature_change': self.settings['temperature_change'],
            'step_size': new_step_size,
            'step_size_change': self.settings['step_size_change']
        }
        return optimise_settings
