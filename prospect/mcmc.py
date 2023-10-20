from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import shutil
from time import time, perf_counter
from types import NoneType
from typing import Any
import numpy as np 
from prospect.input import InputArgument

import sys 

def initialise_mcmc(config_mcmc, kernel, **mcmc_args):
    if config_mcmc.algorithm == 'MetropolisHastings':
        sampler = MetropolisHastings(config_mcmc, kernel, **mcmc_args)
    else:
        raise ValueError('You have specified a non-existing MCMC type.')
    return sampler

"""
    Chain structure

"""

class Chain:
    mults: list[float]
    loglkls: list[float]
    positions: dict[str, np.ndarray] # Keys are parameter names, vals are arrays of positions traversed

    def __init__(self, initial_mults=None, initial_loglkls=None, initial_positions=None) -> None:
        self.mults, self.loglkls, self.positions = [], [], defaultdict(list)
        if initial_mults is not None:
            if initial_loglkls is not None:
                if initial_positions is not None:
                    self.mults += initial_mults
                    self.loglkls += initial_loglkls
                    for param_name, param_vec in initial_positions.items():
                        self.positions[param_name] += list(param_vec)

    def last_varying_position(self, list_of_varying_param_names):
        return {param_name: [self.positions[param_name][-1]] for param_name in list_of_varying_param_names}

    @property
    def last_position(self):
        out = {param: [self.positions[param][-1]] for param in list(self.positions.keys())}
        return out

    @property
    def N(self):
        if len(self.mults) == len(self.loglkls):
                for param_vec in self.positions.values():
                    if len(param_vec) != len(self.mults):
                        break
                else:
                    return len(self.mults)
        raise ValueError('Invalid dimensions in Chain: Non-compatible length of mults, loglkls and positions.')
    
    @property
    def data(self):
        """Returns a [N_points, 2+N_params] numpy array of chain contents in the format of a MontePython chain"""
        pos = np.array(list(self.positions.values()))
        output = np.concatenate([np.stack([self.mults, self.loglkls], axis=0), pos], axis=0).T
        return output

    def __getitem__(self, idx):
        # Return the position at idx
        return {param_name: [param_vec[idx]] for param_name, param_vec in self.positions.items()}
    
    def push_position(self, new_loglkl, new_position):
        self.mults.append(1)
        self.loglkls.append(new_loglkl)
        for param_name, param_vec in self.positions.items():
            param_vec.append(new_position[param_name][0])
    
    def from_indices(self, index_list):
        """Returns a sub-chain of self with the points of the indices given"""
        positions = {}
        for param_name, param_vec in self.positions.items():
            positions[param_name] = np.take(param_vec, index_list)
        return Chain(list(np.take(self.mults, index_list)),
                     list(np.take(self.loglkls, index_list)),
                     positions)

def collapse_chains(chain_list: list[Chain]) -> Chain:
    total_chain = Chain()
    for chain in chain_list:
        total_chain.mults += chain.mults # list addition
        total_chain.loglkls += chain.loglkls
        for param_name, param_vec in chain.positions.items():
            total_chain.positions[param_name] += list(param_vec)
    return total_chain

def compute_covariance_matrix(chain_list: list[Chain]):
    total_chain = collapse_chains(chain_list)
    # bias=True to normalize by N_points, like MP, instead of N_points-1
    # fweights are frequency weights, i.e. multiplicities
    return np.cov(total_chain.data[:, 2:], rowvar=False, bias=True, fweights=total_chain.mults)

"""
    MCMC Algorithms

"""

class BaseMCMC(ABC):
    def __init__(self, config_mcmc, kernel, **mcmc_args):
        super().__init__()
        self.config_mcmc = config_mcmc
        self.kernel = kernel

        if self.config_mcmc.start_from_covmat is not None:
            if type(self.config_mcmc.start_from_covmat) != str:
                self.covmat = self.config_mcmc.start_from_covmat
            else:
                self.covmat = self.kernel.read_covmat(self.config_mcmc.start_from_covmat)
        else:
            self.covmat = self.kernel.get_default_covmat()

        if 'chain' in mcmc_args:
            self.chain = mcmc_args['chain']
        else:
            if self.config_mcmc.start_from_position is not None:
                initial_position = self.kernel.read_initial_position(self.config_mcmc.start_from_position)
            else:
                initial_position = self.kernel.get_default_initial_position()
            self.chain = Chain([1], [self.kernel.loglkl(initial_position)], initial_position)

    @abstractmethod
    def step(self):
        pass

    def run_steps(self, steps_per_iteration: int):
        times = []
        for idx_step in range(steps_per_iteration):
            time_ini = perf_counter()
            self.step()
            time_final = perf_counter()
            times.append(time_final - time_ini)
        print(f"Finished running MCMC for {steps_per_iteration} steps. Average step time is {np.mean(times)} s.")
    
    def run_minutes(self, minutes: float):
        tic = time()
        seconds = minutes*60
        elapsed = 0
        times = []
        while elapsed < seconds:
            time_ini = perf_counter()
            self.step()
            time_final = perf_counter()
            times.append(time_final - time_ini)
            elapsed = time() - tic
        print(f"Finished running MCMC for {minutes} minutes. Average step time is {np.mean(times)} s.")
    
    def finalize(self):
        # Delete montepython log folders
        # shutil.rmtree(self.kernel.mp_dir)
        # Make sure kernel is not dumped when saving
        pass

class MetropolisHastings(BaseMCMC):
    def __init__(self, config_mcmc, kernel, **mcmc_args):
        super().__init__(config_mcmc, kernel, **mcmc_args)

    def step(self):
        prop = self.get_proposal()
        logprior_prop = self.kernel.logprior(prop)
        if logprior_prop == np.inf:
            acceptance = 0
        else:
            loglkl_prop = self.kernel.loglkl(prop)
            # Assume uniform priors => acc. prob. is likelihood ratio
            logpost_current = self.chain.loglkls[-1] + self.kernel.logprior(self.chain.last_position)
            logpost_prop = loglkl_prop + logprior_prop
            # Implicit minus sign on loglkl in our convention
            acceptance = np.exp((logpost_current - logpost_prop)/self.config_mcmc.temperature) 

        rand = np.random.uniform(low=0, high=1)
        if acceptance > rand:
            self.chain.push_position(loglkl_prop, prop)
        else:
            self.chain.mults[-1] += 1
        
    def get_proposal(self):
        # Important to remember fixed values!
        prop = {fixed_param_name: [param_dict['fixed_value']] for fixed_param_name, param_dict in self.kernel.param['fixed'].items()}
        current_means = np.array(list(self.chain.last_varying_position(self.kernel.varying_param_names).values())).flatten()
        # Note that we multiply the covmat by step_size^2 since step_size is a multiplier on the std deviations
        prop_list = np.random.multivariate_normal(current_means, self.covmat*self.config_mcmc.step_size**2)
        for idx, varying_param_name in enumerate(self.kernel.param['varying']):
            prop[varying_param_name] = [prop_list[idx]]
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
        """
            Only implemented on the acceptance criterion, so actual 
            loglkl values are independent of this and can always be compared.
        """
        val_type = float 
        def get_default(self, config_yaml: dict[str, Any]):
            return 1.0
    
    class step_size(InputArgument):
        # Multiplier on the proposal width
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
    
    class start_from_covmat(InputArgument):
        # Specifies the parameter covmat 
        # Specific form depends on which kernel is being used
        # For analytical kernel: Nested lists of numbers defining the matrix
        # For MontePython: string pointing to a '.covmat' file
        # For cobaya: Don't enter!
        # Default value is extracted from the kernel
        val_type = str | list | np.ndarray | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
    
    class start_from_position(InputArgument):
        # Point to start from
        # Specific form depends on which kernel is being used
        # For analytical kernel: List of numbers
        # For MontePython: string pointing to a '.bestfit' file
        # For cobaya: Don't enter!
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
    step_size: step_size
    analyse_automatically: analyse_automatically
    start_from_covmat: start_from_covmat
    start_from_position: start_from_position
