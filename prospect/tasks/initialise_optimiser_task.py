import time
from typing import Type
import numpy as np 
from getdist import loadMCSamples
from prospect.input import Configuration
from prospect.kernels.initialisation import initialise_kernel
from prospect.mcmc import Chain, collapse_chains, compute_covariance_matrix
from prospect.tasks.base_task import BaseTask
from prospect.tasks.optimise_task import OptimiseTask

class InitialiseOptimiserTask(BaseTask):
    """
        This task initialises a profile from a finished MCMC
        The MCMC must be made with the chosen kernel 
        (i.e. cannot start a MontePython-based profile from a cobaya chain)

    """
    priority = 80.0

    def __init__(self, config: Configuration, fixed_param_val: float, repetition_number: int = 0):
        super().__init__(config)
        self.fixed_param_val = fixed_param_val
        self.repetition_number = repetition_number

    def run(self, _):
        kernel = initialise_kernel(self.config.kernel, self.config.io.dir, self.id)
        try:
            profile_param_idx = np.where(np.array(list(kernel.param['varying'].keys())) == self.config.profile.parameter)[0][0]
        except Exception as e:
            print("EXCEPTION:")
            print(f"The requested profile parameter {self.config.profile.parameter} is not recognized by the kernel. Check that your kernel parameter file has the parameter enabled.")
            print("Original exception:")
            raise e
        kernel.set_fixed_parameters({self.config.profile.parameter: self.fixed_param_val})

        tic = time.perf_counter()
        initial_chain = load_mcmc(self.config.profile.start_from_mcmc, kernel, collapse=True)
        print(f"Loaded MCMC {self.config.profile.start_from_mcmc} in {time.perf_counter() - tic:.4} s")

        for param_name in kernel.param['varying'].keys():
            if param_name not in initial_chain.positions.keys():
                # Future: Just ignore this parameter and find a random starting point
                raise KeyError(f"Parameter '{param_name}' not present in the MCMC I'm starting from. Ending initialisation.")

        # Get start_bin_fraction fraction of total points closest to the param_val
        binned_points = get_fraction_of_points_in_bin(initial_chain, self.fixed_param_val, self.config.profile.parameter, self.config.profile.start_bin_fraction)
        
        # Get bestfit among these points
        bestfit_index = np.argmin(binned_points.loglkls)
        self.initial_bestfit = binned_points[bestfit_index]
        self.initial_bestfit[self.config.profile.parameter] = [self.fixed_param_val]
        self.initial_loglkl = kernel.loglkl(self.initial_bestfit)

        # Construct covmat based on these points
        self.initial_covmat = compute_covariance_matrix([binned_points])
        self.initial_covmat = np.delete(self.initial_covmat, [profile_param_idx], 0) # remove row of profile param
        self.initial_covmat = np.delete(self.initial_covmat, [profile_param_idx], 1) # remove column of profile param 

        print(f"Constructed local bestfit and covariance matrix around {self.config.profile.parameter}={self.fixed_param_val}.")

    def emit_tasks(self) -> list[Type[BaseTask]]:
        optimise_settings = {
            'current_best_loglkl': self.initial_loglkl,
            'fixed_param_val': self.fixed_param_val,
            'initial_position': self.initial_bestfit,
            'covmat': self.initial_covmat,
            'temperature': self.config.profile.temperature_range[0],
            'temperature_change': self.get_temperature_change(),
            'step_size': self.config.profile.step_size_range[0],
            'step_size_change': self.get_step_size_change(),
            'iteration_number': 0,
            'repetition_number': self.repetition_number
        }
        return [OptimiseTask(self.config, optimise_settings)]

    def get_temperature_change(self):
        temp_change = 0.
        if self.config.profile.temperature_schedule == 'exponential':
            # The temperature change parameter is the rate, i.e. the number that
            # the temperature is multiplied by at each iteration
            temp_change = (self.config.profile.temperature_range[1]/self.config.profile.temperature_range[0])**(1/self.config.profile.max_iterations)
        else:
            raise ValueError('Invalid temperature schedule.')
        return temp_change

    def get_step_size_change(self):
        step_size_change = 0.
        if self.config.profile.step_size_schedule == 'exponential':
            # Same logic as in `get_temperature_change()`
            step_size_change = (self.config.profile.step_size_range[1]/self.config.profile.step_size_range[0])**(1/self.config.profile.max_iterations)
        else:
            raise ValueError('Invalid step size schedule.')
        return step_size_change

"""
    Functions for estimating profile likelihoods from MCMC chains


"""

def load_mcmc(mcmc_path: str, kernel, collapse=False) -> list[Chain]:
    """Loads the contents of an unpacked MCMC into a list of chains."""
    samples = loadMCSamples(mcmc_path)
    param_name_list = [param_name.name for param_name in samples.getParamNames().names]

    def create_prospect_chain(getdist_samples, param_name_list):
        prospect_chain = Chain()
        prospect_chain.mults = getdist_samples.weights
        prospect_chain.loglkls = getdist_samples.loglikes
        for idx_param, param_name in enumerate(param_name_list):
            prospect_chain.positions[param_name] = getdist_samples.samples[:, idx_param]
        return prospect_chain

    if collapse:
        return create_prospect_chain(samples, param_name_list)
    else:
        getdist_chains = samples.getSeparateChains()
        prospect_chains = []
        for getdist_chain in getdist_chains:
            prospect_chains.append(create_prospect_chain(getdist_chain, param_name_list))
        return prospect_chains

def get_points_in_bin(chain: Chain, param_val, param_name, N_points):
    """
        Returns a sub-chain of chain input consisting of the 'N_points' points 
        closest to the value 'param_val' of the parameter 'param_name'
    """
    # Sort points by their absolute distance to param_val
    distance_points = np.abs(np.array(chain.positions[param_name]) - param_val)
    ordered_indices = np.argsort(distance_points)
    # Take the first N_points points
    output_indices = ordered_indices[:N_points]
    return chain.from_indices(output_indices)

def get_fraction_of_points_in_bin(chain, param_val, param_name, fraction):
    N_points = int(np.ceil(fraction*chain.N))
    return get_points_in_bin(chain, param_val, param_name, N_points)

def estimate_bestfit(chains, param_val, param_idx, bin_fraction=0.1):
    binned_points = get_fraction_of_points_in_bin(chains, param_val, param_idx, bin_fraction)
    bestfit_index = np.argmin(binned_points[:, 1])
    return binned_points[bestfit_index]