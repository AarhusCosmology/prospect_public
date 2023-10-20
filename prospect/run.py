from dataclasses import dataclass
import os
import sys
from types import NoneType
from typing import Any
from prospect.communication import Scheduler
from prospect.io import prepare_run, read_user_input, load_config, load_state
from prospect.input import Configuration, InputArgument

def master_process(config_yaml) -> bool:
    if config_yaml['run']['mode'] == 'mpi':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank != 0:
            return False
    return True

def run(user_inp: str) -> None:
    config_yaml, resume = read_user_input(user_inp)
    if resume:
        config = load_config(user_inp)
        config.io.dir = config_yaml['io']['dir']
    else:
        config = Configuration(config_yaml)

    if config.run.mode == 'mpi':
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            raise ValueError('You cannot run PROSPECT using MPI with only one process.')
        pool = MPICommExecutor
        pool_kwargs = {'comm': comm, 'root': 0}
    elif config.run.mode == 'threaded':
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor
        pool_kwargs = {'max_workers': config.run.num_processes}
    elif config.run.mode == 'serial':
        from prospect.communication import SerialContext
        pool = SerialContext
        pool_kwargs = {}
    else:
        raise ValueError(f"Run mode '{config.run.mode}' not recognized. Choose either 'mpi', 'threaded' or 'serial'.")

    with pool(**pool_kwargs) as executor:
        if executor is not None:
            print(f"Running PROSPECT with mode *{config.run.mode}* on {config.run.num_processes} processes...")
            if resume:
                print(f"Resuming PROSPECT snapshop in {user_inp}.")
                state = load_state(user_inp)
            else:
                print(f"Starting PROSPECT from input file {user_inp}.")
                prepare_run(config)
                state = False
            scheduler = Scheduler(config, state)
            scheduler.delegate(executor)
            scheduler.finalize(executor)

    if config.run.mode == 'mpi':
        comm.Barrier()
        MPI.Finalize()
    sys.exit(0)

def run_from_shell() -> None:
    """
        Wrapper for run() using a setuptools entry point
        Allows running from command-line with the command `prospect input/test.yaml`
        Only takes arguments using argparse

    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='+')
    args = parser.parse_args()
    run(args.input_file[0])

def load(prospect_folder) -> Scheduler:
    """
        Loads a scheduler with all the information content of the 
        PROSPECT snapshot in prospect_pkl. Intended for use in interactive sessions.

        prospect_folder: 
            A folder with a previous prospect run.
            Must contain at least a snapshot.pkl and log.yaml.
    
    """
    config = load_config(prospect_folder)
    config.io.dir = prospect_folder
    state = load_state(prospect_folder)
    return Scheduler(config, state)

def analyse(prospect_folder):
    snapshot = load(prospect_folder)
    if snapshot.config.run.jobtype == 'profile':
        print(f"Analysing {prospect_folder} as a profile likelihood run.")
        analysis_task = snapshot.get_profile_analysis()

    elif snapshot.config.run.jobtype == 'mcmc':
        print(f"Analysing {prospect_folder} as an MCMC run.")
        from prospect.tasks.mcmc_task import AnalyseMCMCTask
        # Pick out the unique task of largest id belonging to each chain
        final_tasks = {}
        for chain_id, mcmc_task in {task.mcmc_args['chain_id']: task for task in snapshot.tasks.done.values() if task.type == 'MCMCTask'}:
            if chain_id not in final_tasks or mcmc_task.id > final_tasks[chain_id].id:
                final_tasks[chain_id] = mcmc_task
        snapshot.config.mcmc.analyse_automatically = True
        analysis_task = AnalyseMCMCTask(snapshot.config, required_task_ids=[task.id for task in final_tasks.values()])
        
    else:
        raise KeyError(f'Jobtype {snapshot.config.run.jobtype} not understood.')
    
    analysis_task.run([task for task in snapshot.tasks.done.values() if task.id in analysis_task.required_task_ids])
    print(f"\nFinished analysing {prospect_folder}. Enjoy!")

def analyse_from_shell():
    """
        Analyses the prospect folder if it is a profile job.
    
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('prospect_folder', nargs='+')
    args = parser.parse_args()
    analyse(args.prospect_folder[0])


"""
    Definition of user arguments related to run.py
"""

@dataclass
class Arguments:
    class jobtype(InputArgument):
        val_type = str
        allowed_values = ['mcmc', 'profile']
    
    class mode(InputArgument):
        val_type = str
        default = 'threaded'
        allowed_values = ['mpi', 'threaded', 'serial']
    
    class num_processes(InputArgument):
        val_type = int
        def get_default(self, config_yaml):
            if config_yaml['run']['mode'] == 'mpi':
                from mpi4py import MPI
                return MPI.COMM_WORLD.Get_size()
            elif config_yaml['run']['mode'] == 'threaded':
                return os.cpu_count() or 1
            elif config_yaml['run']['mode'] == 'serial':
                return 1
    
    class numpy_random_seed(InputArgument):
        val_type = float | int | NoneType
        def get_default(self, config_yaml: dict[str, Any]):
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['numpy_random_seed'] is not None:
                import numpy as np 
                np.random.seed(config['numpy_random_seed'])

    jobtype: jobtype
    mode: mode
    num_processes: num_processes
    numpy_random_seed: numpy_random_seed
