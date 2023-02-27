from abc import ABC, abstractmethod
import numpy as np 

def initialize_mcmc(config_mcmc, kernel):
    if config_mcmc['algorithm'] == 'MetropolisHastings':
        sampler = MetropolisHastings(config_mcmc, kernel)
    else:
        raise ValueError('You have specified a non-existing MCMC type.')
    return sampler

class BaseMCMC(ABC):
    def __init__(self, config_mcmc, kernel, initial_position=None):
        super().__init__()
        self.config_mcmc = config_mcmc
        self.kernel = kernel
        
        if initial_position != None:
            self.positions = [initial_position]
        else:
            self.positions = [np.zeros([self.kernel.dimension])]

        # Here we might need to do MontePython's trick when starting
        self.loglkls = [self.kernel.loglkl(self.positions[-1])]
        self.mults = [1]

        if 'temperature' in config_mcmc:
            self.temp = config_mcmc['temperature']
        else:
            self.temp = 1.0
    
    @abstractmethod
    def step(self):
        pass

    def loglkl(self, position): # Is this the best way to add temp? Unsafe if writing self.kernel.loglkl in the code...
        return self.kernel.loglkl(position)/self.temp

    def chain(self, N_steps):
        for idx_step in range(N_steps):
            self.step()

class MetropolisHastings(BaseMCMC):
    def __init__(self, config_mcmc, kernel, initial_position=None):
        super().__init__(config_mcmc, kernel, initial_position)
        if not 'covmat' in config_mcmc:
            raise ValueError('You must give a covariance matrix for the Metropolis Hastings proposal density.')
        else:
            self.covmat = self.config_mcmc['covmat']

    def step(self):
        prop = np.random.multivariate_normal(self.positions[-1], self.covmat)
        loglkl_prop = self.loglkl(prop)
        alpha = loglkl_prop/self.loglkls[-1] # Assume uniform priors => acc. prob. is likelihood ratio 
        acceptance = min(1, alpha) 
        # Enforce uniform prior bound 
        for idx_param, prior_bound in enumerate(self.kernel.config_kernel['prior']):
            if prop[idx_param] < prior_bound[0] or prop[idx_param] > prior_bound[1]:
                acceptance = 0
        rand = np.random.uniform(low=0, high=1)
        if acceptance > rand:
            self.positions.append(prop)
            self.loglkls.append(loglkl_prop)
            self.mults.append(1)
        else:
            self.mults[-1] += 1


