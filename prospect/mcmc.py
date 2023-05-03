import numpy as np 
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from prospect.input import InputArgument

def initialise_mcmc(config_mcmc, kernel, **mcmc_args):
    if config_mcmc.algorithm == 'MetropolisHastings':
        sampler = MetropolisHastings(config_mcmc, kernel, **mcmc_args)
    else:
        raise ValueError('You have specified a non-existing MCMC type.')
    return sampler

@dataclass
class Chain:
    mults: list[float]
    loglkls: list[float]
    positions: list[np.ndarray]

    def append(self, other_chain):
        if other_chain.positions[0].shape[0] != self.positions[0].shape[0]:
            raise ValueError(f'Chains with different amount of parameters cannot be appended.')
        self.mults += other_chain.mults
        self.loglkls += other_chain.loglkls
        self.positions += other_chain.positions

class BaseMCMC(ABC):
    def __init__(self, config_mcmc, kernel, **mcmc_args):
        super().__init__()
        self.config_mcmc = config_mcmc
        self.kernel = kernel
        self.temp = config_mcmc.temperature

        if 'chain' in mcmc_args:
            self.chain = mcmc_args['chain']
        else:
            if 'initial_position' in mcmc_args:
                initial_position = mcmc_args['initial_position']
            else:
                # Come up with a (random) initial position!
                # Here we might need to do MontePython's trick when starting
                initial_position = np.zeros([self.kernel.dimension])
            self.chain = Chain([1], [self.loglkl(initial_position)], [initial_position])

    @abstractmethod
    def step(self):
        pass

    def loglkl(self, position): # Is this the best way to add temp? Unsafe if writing self.kernel.loglkl in the code...
        return self.kernel.loglkl(position)/self.temp

    def run_chain(self, N_steps):
        for idx_step in range(N_steps):
            self.step()

class MetropolisHastings(BaseMCMC):
    def __init__(self, config_mcmc, kernel, **mcmc_args):
        super().__init__(config_mcmc, kernel, **mcmc_args)
        self.covmat = self.config_mcmc.covmat

    def step(self):
        prop = np.random.multivariate_normal(self.chain.positions[-1], self.covmat)
        loglkl_prop = self.loglkl(prop)
        # Assume uniform priors => acc. prob. is likelihood ratio 
        alpha = self.chain.loglkls[-1]/loglkl_prop # Flipped fraction since loglkl = minus_loglkl in our convention
        acceptance = min(1, alpha) 
        # Enforce uniform prior bound 
        for idx_param, (param_name, prior_bound) in enumerate(self.kernel.param['param_dict'].items()):
            if prop[idx_param] < prior_bound[0] or prop[idx_param] > prior_bound[1]:
                acceptance = 0
        rand = np.random.uniform(low=0, high=1)
        if acceptance > rand:
            self.chain.positions.append(prop)
            self.chain.loglkls.append(loglkl_prop)
            self.chain.mults.append(1)
        else:
            self.chain.mults[-1] += 1


"""
    Definition of user arguments related to mcmc.py
"""

@dataclass
class Arguments:
    class algorithm(InputArgument):
        val_type = str
        allowed_values = ['MetropolisHastings']
        default = 'MetropolisHastings'

    class N_chains(InputArgument):
        val_type = int
    
    class N_steps(InputArgument): # Replace with convergence_Rm1 when merging with #5
        val_type = int 
    
    class convergence_Rm1(InputArgument):
        val_type = float
    
    class covmat(InputArgument):
        val_type = list | np.ndarray
        default = None # Can make a smart default later
    
    class unpack_at_dump(InputArgument):
        val_type = bool
        default = True
    
    class temperature(InputArgument):
        val_type = float 
        
        def get_default(self, config_yaml: dict[str, Any]):
            return 1.0
    
    class analyse_automatically(InputArgument):
        allowed_values = [True, False, 'yes', 'y']
        def get_default(self, config_yaml: dict[str, Any]):
            return False
        def validate(self, config: dict[str, Any]) -> None:
            if not config['unpack_at_dump']:
                raise ValueError("Must unpack MCMC in order to analyse; please set 'unpack_at_dump' to True.")
    
    algorithm: algorithm
    N_chains: N_chains
    N_steps: N_steps
    convergence_Rm1: convergence_Rm1
    covmat: covmat
    unpack_at_dump: unpack_at_dump
    temperature: temperature
    analyse_automatically: analyse_automatically
