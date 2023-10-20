from copy import deepcopy
from dataclasses import dataclass
from types import NoneType
from typing import Any
import numpy as np 
import os
import yaml
from prospect.communication import Scheduler
from prospect.input import InputArgument, Configuration
from prospect.run import load, load_state, load_config
from prospect.tasks.base_task import BaseTask
from prospect.tasks.analyse_profile_task import AnalyseProfileTask
from prospect.tasks.initialise_optimiser_task import InitialiseOptimiserTask, get_initial_step_size_change, get_temperature_change
from prospect.tasks.optimise_task import OptimiseTask


def reanneal(yaml_schedule, prospect_folder, override_queue=False):
    """
        Adds new OptimiseTasks to the queue, according to the given yaml file, and resumes.
            - yaml_schedule should point to a yaml file whose contents in dict-form are 
              {'profile': {dict with schedule specifications}}

            - prospect_folder should be the path to an existing prospect run



        Currently, the only full schedule specification should contain:
            optimiser, temperature_schedule, temperature_range,
            step_size_schedule, step_size_range,
            minutes_per_iteration OR steps_per_iteration,
            max_iterations, repetitions. 
        In principle, any argument inside the profile module can be changed, but changing
        other arguments than the above is untested.
    
    """
    new_schedule = yaml.full_load(open(yaml_schedule, 'r'))
    if not 'profile' in new_schedule:
        raise KeyError("Reanneal yaml file must contain 'profile' as the first and only key.")
    
    snapshot = load(prospect_folder)
    new_config_dict = deepcopy(snapshot.config.config_dict)
    for key, val in new_schedule['profile'].items():
        new_config_dict['profile'][key] = val
    new_config = Configuration(new_config_dict)
    new_config.io.dir = os.path.join(os.getcwd(), prospect_folder)

    if snapshot.tasks.ongoing or snapshot.tasks.unready or snapshot.tasks.queued:
        if override_queue:
            # WARNING: Removes all non-finised tasks. Note: ongoing tasks should already
            # have been moved to the ready queue upon loading.
            print(f"WARNING: Reannealing with override=True. Deleting all unfinished tasks in snapshot {prospect_folder}.")
            snapshot.tasks.ongoing = {}
            snapshot.tasks.unready = {}
            snapshot.tasks.queued = []
        else:
            raise ValueError('Cannot reanneal a run that did not finish. Please resume the run before reannealing.')

    scheduler = Scheduler(new_config, snapshot.tasks)

    # Create a dict with keys of param_vals and values that hold the id of the best task and its minimum loglkl
    optimisetasks = [task for task in snapshot.tasks.done.values() if task.type == 'OptimiseTask']
    best_tasks = {}
    for task in optimisetasks:
        if task.optimise_settings['fixed_param_val'] not in best_tasks:
            best_tasks[task.optimise_settings['fixed_param_val']] = {'id': task.id, 'loglkl': task.optimiser.bestfit['loglkl']}
        elif best_tasks[task.optimise_settings['fixed_param_val']]['loglkl'] > task.optimiser.bestfit['loglkl']:
            best_tasks[task.optimise_settings['fixed_param_val']] = {'id': task.id, 'loglkl': task.optimiser.bestfit['loglkl']}
    
    # Push new tasks
    submit_count = 0
    for param_val in new_config.profile.values:
        if param_val in best_tasks:
            # Start from the old bestfit
            for idx_rep in range(new_config.profile.repetitions):
                optimise_settings = {
                    'current_best_loglkl': best_tasks[param_val]['loglkl'],
                    'fixed_param_val': param_val,
                    'initial_position': snapshot.tasks.done[best_tasks[param_val]['id']].optimiser.bestfit['position'],
                    'covmat': snapshot.tasks.done[best_tasks[param_val]['id']].optimiser.covmat,
                    'temperature': new_config.profile.temperature_range[0],
                    'temperature_change': get_temperature_change(new_config),
                    'step_size': new_config.profile.step_size_range[0],
                    'step_size_change': get_initial_step_size_change(new_config),
                    'iteration_number': snapshot.tasks.done[best_tasks[param_val]['id']].optimiser.settings['iteration_number'] + 1,
                    'repetition_number': idx_rep
                }
                scheduler.push_task(OptimiseTask(scheduler.config, optimise_settings))
                submit_count += 1
        else:
            # Launch new optimisation
            for idx_rep in range(new_config.profile.repetitions):
                scheduler.push_task(InitialiseOptimiserTask(scheduler.config, param_val, idx_rep))
                submit_count += 1
    
    scheduler.dump_snapshot()
    scheduler.status_update()
    print(f"Added {submit_count} optimisation tasks to snapshot {prospect_folder} with the schedule from {yaml_schedule}.")
    print(f"You may now resume the snapshot to run the tasks.")

def reanneal_from_shell():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', metavar='YAML', type=str, help="Path to a .yaml with the new schedule.")
    parser.add_argument('-o', metavar='FOLDER', type=str, help="Path to a PROSPECT folder to reanneal.")
    parser.add_argument('--override', help="Delete any unfinished tasks in the PROSPECT snapshot.", action='store_true')

    args = parser.parse_args()
    reanneal(args.y, args.o, args.override)

def load_profile(prospect_folder, direct_txt=False):
    if direct_txt:
        # If direct_txt is True, prospect_folder must point to the .txt file containing the results directly
        data_file = prospect_folder
    else:
        config = load_config(prospect_folder)
        data_file = f'{prospect_folder}/profile/{config.profile.parameter}.txt'
    if not os.path.isfile(data_file):
        raise KeyError(f"Could not load profile likelihood results: '{data_file}' not found.")
    results = np.genfromtxt(data_file, delimiter="\t", dtype=str, skip_header=0, autostrip=True)
    res_dict = {}
    for idx, entry in enumerate(results[0]):
        res_dict[entry] = np.array([float(val.strip('[]')) for val in results[1:, idx]])
    return res_dict
    

"""
    Definition of user arguments related to mcmc.py
"""

@dataclass
class Arguments:
    class parameter(InputArgument):
        val_type = str
    
    class values(InputArgument):
        """Parameter values where the profile is evaluated"""
        val_type = list[float] | list[int] | np.ndarray | list
    
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
        allowed_values = ['exponential', 'adaptive']
        def get_default(self, config_yaml: dict[str, Any]):
            return 'adaptive'
        def validate(self, config: dict[str, Any]) -> None:
            assert config['optimiser'] == 'simulated annealing'
    
    class step_size_range(InputArgument):
        val_type = list[float] | list[np.float64] | np.ndarray | list | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            None
        def validate(self, config: dict[str, Any]) -> None:
            if config['step_size_schedule'] == 'exponential':
                if config['step_size_range'] is None:
                    raise ValueError("Input 'step_size_range' is set to None, but must be set when using the 'exponential' schedule type.")

    class step_size_adaptive_interval(InputArgument):
        # The acceptance rate interval which will be aimed for
        val_type = list[float] | list[np.float64] | np.ndarray | list | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return [0.05, 0.15]

    class step_size_adaptive_multiplier(InputArgument):
        """
            The step size multiplier rate r at each iteration in the adaptive annealing
            i.e. if AR > adaptive_interval:
                new_ar = old_ar*(1 + r)
            elif AR < adaptive_interval:
                new_ar = old_ar*(1 - r)

            If 'adaptive', it will use estimate the linear response on AR to find the best value
            at each iteration
        """
        val_type = float | np.float64 | str
        def get_default(self, config_yaml: dict[str, Any]):
            return 'adaptive'
        def validate(self, config: dict[str, Any]) -> None:
            if isinstance(config['step_size_adaptive_multiplier'], str):
                if not config['step_size_adaptive_multiplier'] == 'adaptive':
                    raise ValueError("The only allowed string input for 'step_size_adaptive_multiplier' is 'adaptive'.'")
    
    class step_size_adaptive_initial(InputArgument):
        # Initial step size when using adaptive step size
        val_type = float | np.float64
        def get_default(self, config_yaml: dict[str, Any]):
            return 1.0
        def validate(self, config: dict[str, Any]) -> None:
            if config['step_size_schedule'] == 'adaptive':
                # Set initial step size variable in the range parameter
                config['step_size_range'] = [config['step_size_adaptive_initial'], None]

    class max_iterations(InputArgument):
        val_type = int | float
    
    class repetitions(InputArgument):
        val_type = int | float
        def get_default(self, config_yaml: dict[str, Any]):
            return 1

    class start_from_mcmc(InputArgument):
        val_type = str | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
    
    class start_bin_fraction(InputArgument):
        """
           Fraction of points (wrt. total amount in MCMC)
           around the profile parameter values included in the 
           initial profile generation
    
        """
        val_type = float
        def get_default(self, config_yaml: dict[str, Any]):
            return 0.1

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
    
    class start_from_profile(InputArgument):
        # .txt file with content loadable by load_profile
        # If the variables match, this starts the profile from the old profile
        # such that for each fixed param value in the new profile, the closest
        # bestfit in the old profile is taken as the starting point
        # If inputting this, remember to also put either start_from_mcmc or 
        # start_from_covmat in order to also get a covmat
        val_type = str | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['start_from_profile'] is not None:
                assert config['start_from_mcmc'] is not None or config['start_from_covmat'] is not None
    
        
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
    step_size_adaptive_interval: step_size_adaptive_interval
    step_size_adaptive_multiplier: step_size_adaptive_multiplier
    step_size_adaptive_initial: step_size_adaptive_initial
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
    start_from_covmat: start_from_covmat
    start_from_position: start_from_position
    start_from_profile: start_from_profile
    


