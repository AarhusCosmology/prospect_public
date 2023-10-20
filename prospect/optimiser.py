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
            'acceptance_rate': len(self.mcmc.chain.loglkls)/np.sum(self.mcmc.chain.mults),
            'accepted_steps': len(self.mcmc.chain.loglkls),
            'iteration_number': self.settings['iteration_number'],
            'position': {}
        }
        for param_name, param_vec in self.mcmc.chain.positions.items():
            self.bestfit['position'][param_name] = [param_vec[bestfit_idx]]
        #print(f"Optimiser set bestfit loglkl={self.bestfit['loglkl']}. Chain length is {len(self.mcmc.chain.loglkls)}.")
    
    def get_next_iteration_settings(self):
        if self.config.profile.temperature_schedule == 'exponential':
            new_temp = self.settings['temperature']*self.settings['temperature_change']
        else:
            raise ValueError('Invalid temperature schedule.')
        if self.config.profile.step_size_schedule == 'exponential':
            new_step_size = self.settings['step_size']*self.settings['step_size_change']
        elif self.config.profile.step_size_schedule == 'adaptive':
            if self.config.profile.step_size_adaptive_multiplier == 'adaptive':
                if not 'previous_multiplier' in self.settings['step_size_change']:
                    # First iteration, cannot use history to set multiplier
                    # Hardcoded initial multiplier
                    if self.bestfit['acceptance_rate'] > self.config.profile.step_size_adaptive_interval[1]:
                        initial_multiplier = 0.05
                    elif self.bestfit['acceptance_rate'] < self.config.profile.step_size_adaptive_interval[0]:
                        initial_multiplier = -0.05
                    new_step_size = self.settings['step_size']*(1 + initial_multiplier)
                    self.settings['step_size_change']['previous_multiplier'] = initial_multiplier
                    self.settings['step_size_change']['previous_acceptance_rate'] = self.bestfit['acceptance_rate']
                elif not 'current_multiplier' in self.settings['step_size_change']:
                    # Second iteration
                    if self.bestfit['acceptance_rate'] > self.config.profile.step_size_adaptive_interval[1]:
                        new_multiplier = 0.1
                    elif self.bestfit['acceptance_rate'] < self.config.profile.step_size_adaptive_interval[0]:
                        new_multiplier = -0.1
                    new_step_size = self.settings['step_size']*(1 + new_multiplier)
                    self.settings['step_size_change']['current_multiplier'] = new_multiplier
                    self.settings['step_size_change']['current_acceptance_rate'] = self.bestfit['acceptance_rate']
                else:
                    # At least third iteration, can now use estimated linear response 
                    # on AR to set optimal multiplier targetting the middle point of the desired AR interval
                    AR_desired = np.mean(self.config.profile.step_size_adaptive_interval)
                    AR_current = self.bestfit['acceptance_rate']
                    AR_previous = self.settings['step_size_change']['previous_acceptance_rate']
                    m_current = self.settings['step_size_change']['current_multiplier']
                    m_previous = self.settings['step_size_change']['previous_multiplier']
                    m_new = (AR_desired - AR_current)/(AR_current - AR_previous)*(m_current - m_previous) + m_current
                    #m_new = (AR_desired - AR_current)/(AR_current - AR_previous)*m_current/(1 + m_current)

                    # For stability
                    m_new = min(m_new, 1.0)
                    m_new = max(-0.9, m_new)
                    
                    new_step_size = self.settings['step_size']*(1 + m_new)
                    
                    # Update for next iteration
                    self.settings['step_size_change'] = {
                        'previous_acceptance_rate': AR_current,
                        'current_multiplier': m_new,
                        'previous_multiplier': m_current
                    }
            else:
                if self.bestfit['acceptance_rate'] > self.config.profile.step_size_adaptive_interval[1]:
                    # Acceptance rate is large, increase step size
                    new_step_size = self.settings['step_size']*(1 + self.config.profile.step_size_adaptive_multiplier)
                elif self.bestfit['acceptance_rate'] < self.config.profile.step_size_adaptive_interval[0]:
                    # Acceptance rate is small, reduce step size
                    new_step_size = self.settings['step_size']*(1 - self.config.profile.step_size_adaptive_multiplier)
                
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
