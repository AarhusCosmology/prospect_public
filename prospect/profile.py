from dataclasses import dataclass
import os
from types import NoneType
from typing import Any
import numpy as np 
from prospect.input import InputArgument

"""
    Definition of user arguments related to mcmc.py
"""

@dataclass
class Arguments:
    class parameter(InputArgument):
        val_type = str
    
    class values(InputArgument):
        """Parameter values where the profile is evaluated"""
        val_type = list[float] | list[np.float64] | np.ndarray | list
    
    class optimiser(InputArgument):
        allowed_values = ['simulated annealing']
        def get_default(self, config_yaml: dict[str, Any]):
            return 'simulated annealing'

    class temperature_schedule(InputArgument):
        allowed_values = ['exponential']
        def get_default(self, config_yaml: dict[str, Any]):
            return 'exponential'
        def validate(self, config: dict[str, Any]) -> None:
            assert config['optimiser'] == 'simulated annealing'
    
    class temperature_range(InputArgument):
        val_type = list[float] | list[np.float64] | np.ndarray | list
    
    class step_size_schedule(InputArgument):
        allowed_values = ['exponential']
        def get_default(self, config_yaml: dict[str, Any]):
            return 'exponential'
        def validate(self, config: dict[str, Any]) -> None:
            assert config['optimiser'] == 'simulated annealing'
    
    class step_size_range(InputArgument):
        val_type = list[float] | list[np.float64] | np.ndarray | list
    
    class max_iterations(InputArgument):
        val_type = int | float
    
    class repetitions(InputArgument):
        val_type = int | float
        def get_default(self, config_yaml: dict[str, Any]):
            return 1

    class start_from_mcmc(InputArgument):
        val_type = str
    
    class start_bin_fraction(InputArgument):
        """
           Fraction of points (wrt. total amount in MCMC)
           around the profile parameter values included in the 
           initial profile generation
    
        """
        val_type = float
        def get_default(self, config_yaml: dict[str, Any]):
            return 0.1
        
    class chi2_tolerance(InputArgument):
        """
            Used as convergence criterion for the iterative optimisers:
            If chi2 changed less than this since last iteration, stop.
        
        """
        val_type = float | int
        def get_default(self, config_yaml: dict[str, Any]):
            return 0.05
    
    class detailed_plot(InputArgument):
        """
            If true, plots initial points and each rep in profile
            Also, if using analytical kernel, plots the analytical profile

        """
        val_type = bool
        def get_default(self, config_yaml: dict[str, Any]):
            return False
    
    class plot_Delta_chi2(InputArgument):
        val_type = bool
        def get_default(self, config_yaml: dict[str, Any]):
            return True
    
    class plot_profile(InputArgument):
        val_type = bool
        def get_default(self, config_yaml: dict[str, Any]):
            return True

    class plot_schedule(InputArgument):
        val_type = bool
        def get_default(self, config_yaml: dict[str, Any]):
            return False
    
    class confidence_levels(InputArgument):
        # Computes confidence intervals at these levels
        val_type = list
        def get_default(self, config_yaml: dict[str, Any]):
            return [0.68, 0.95]

    parameter: parameter
    values: values
    optimiser: optimiser
    temperature_schedule: temperature_schedule
    temperature_range: temperature_range
    step_size_schedule: step_size_schedule
    step_size_range: step_size_range
    #from prospect.mcmc import Arguments as mcmc_args
    #steps_per_iteration: mcmc_args.steps_per_iteration
    #minutes_per_iteration: mcmc_args.minutes_per_iteration
    """NOTE: The above doesn't work, so we duplicate code temporarily."""
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
    steps_per_iteration: steps_per_iteration
    minutes_per_iteration: minutes_per_iteration

    max_iterations: max_iterations
    repetitions: repetitions
    start_from_mcmc: start_from_mcmc
    start_bin_fraction: start_bin_fraction
    chi2_tolerance: chi2_tolerance
    plot_profile: plot_profile
    detailed_plot: detailed_plot
    plot_Delta_chi2: plot_Delta_chi2
    plot_schedule: plot_schedule
    confidence_levels: confidence_levels
    


