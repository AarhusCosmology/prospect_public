from abc import ABC, abstractmethod
from dataclasses import dataclass
import shutil
from time import time
from types import NoneType
from typing import Any
import numpy as np 
from prospect.input import InputArgument

def initialise_mcmc(config_mcmc, kernel, **mcmc_args):
    if config_mcmc.algorithm == 'MetropolisHastings':
        sampler = MetropolisHastings(config_mcmc, kernel, **mcmc_args)
    else:
        raise ValueError('You have specified a non-existing MCMC type.')
    return sampler

class Chain:
    mults: list[float]
    loglkls: list[float]
    positions: dict[str, np.ndarray] # Keys are parameter names, vals are arrays of positions traversed
    def __init__(self, initial_mult, initial_loglkl, initial_position) -> None:
        self.mults = [initial_mult]
        self.loglkls = [initial_loglkl]
        self.positions = {param: [val] for param, val in initial_position.items()}

    @property
    def last_position(self):
        out = {param: self.positions[param][-1] for param in list(self.positions.keys())}
        return out
    
    def push_position(self, new_loglkl, new_position):
        self.mults.append(1)
        self.loglkls.append(new_loglkl)
        for param_name, param_vec in self.positions.items():
            param_vec.append(new_position[param_name])

class BaseMCMC(ABC):
    def __init__(self, config_mcmc, kernel, **mcmc_args):
        super().__init__()
        self.config_mcmc = config_mcmc
        self.kernel = kernel
        self.temp = config_mcmc.temperature

        if self.config_mcmc.covmat is not None:
            self.covmat = self.kernel.get_covmat(self.config_mcmc.covmat)
        else:
            self.covmat = self.kernel.get_default_covmat()

        if 'chain' in mcmc_args:
            self.chain = mcmc_args['chain']
        else:
            if self.config_mcmc.initial_position is not None:
                initial_position = self.kernel.get_initial_position(self.config_mcmc.initial_position)
            else:
                initial_position = self.kernel.get_default_initial_position()
            self.chain = Chain(1, self.loglkl(initial_position), initial_position)

    @abstractmethod
    def step(self):
        pass

    def loglkl(self, position): # Is this the best way to add temp? Unsafe if writing self.kernel.loglkl in the code...
        return self.kernel.loglkl(position)/self.temp

    def run_steps(self, steps_per_iteration: int):
        for idx_step in range(steps_per_iteration):
            self.step()
    
    def run_minutes(self, minutes: float):
        tic = time()
        seconds = minutes*60
        elapsed = 0
        while elapsed < seconds:
            self.step()
            elapsed = time() - tic
    
    def finalize(self):
        # Delete montepython log folders
        shutil.rmtree(self.kernel.mp_dir)
        # Make sure kernel is not dumped when saving
        del self.kernel

class MetropolisHastings(BaseMCMC):
    def __init__(self, config_mcmc, kernel, **mcmc_args):
        super().__init__(config_mcmc, kernel, **mcmc_args)

    def step(self):
        prop = self.get_proposal()
        loglkl_prop = self.loglkl(prop)
        # Assume uniform priors => acc. prob. is likelihood ratio 
        alpha = self.chain.loglkls[-1]/loglkl_prop # Flipped fraction since loglkl = minus_loglkl in our convention
        acceptance = min(1, alpha) 
        # Enforce uniform prior bound 
        for param_name, param_val in prop.items():
            prior = self.kernel.param[param_name]['prior']
            if prior[0] is not None:
                if param_val < prior[0]:
                    acceptance = 0
            if prior[1] is not None:
                if param_val > prior[1]:
                    acceptance = 0
                
        rand = np.random.uniform(low=0, high=1)
        if acceptance > rand:
            self.chain.push_position(loglkl_prop, prop)
        else:
            self.chain.mults[-1] += 1
        
    def get_proposal(self):
        current_means = list(self.chain.last_position.values())
        prop_list = np.random.multivariate_normal(current_means, self.covmat)
        prop = {param_name: prop_list[idx] for idx, param_name in enumerate(list(self.kernel.param.keys()))}
        return prop

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
    
    class steps_per_iteration(InputArgument):
        val_type = int | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['steps_per_iteration'] is not None:
                assert config['minutes_per_iteration'] is None
    
    class minutes_per_iteration(InputArgument):
        val_type = int | float | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['minutes_per_iteration'] is not None:
                assert config['minutes_per_iteration'] > 0
                assert config['steps_per_iteration'] is None
    
    class convergence_Rm1(InputArgument):
        val_type = float
    
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
    
    class covmat(InputArgument):
        # Specifies the parameter covmat 
        # Specific form depends on which kernel is being used
        # For analytical kernel: Nested lists of numbers defining the matrix
        # For MontePython: string pointing to a '.covmat' file
        # For cobaya: TBA
        # Default value is extracted from the kernel
        val_type = str | list | np.ndarray | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
    
    class initial_position(InputArgument):
        # Point to start from
        # Specific form depends on which kernel is being used
        # For analytical kernel: List of numbers
        # For MontePython: string pointing to a '.bestfit' file
        # For cobaya: TBA
        # Default value is extracted from the kernel
        val_type = str | list | np.ndarray | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
    
    algorithm: algorithm
    N_chains: N_chains
    steps_per_iteration: steps_per_iteration
    minutes_per_iteration: minutes_per_iteration
    convergence_Rm1: convergence_Rm1
    unpack_at_dump: unpack_at_dump
    temperature: temperature
    analyse_automatically: analyse_automatically
    covmat: covmat
    initial_position: initial_position
