from dataclasses import dataclass
import os
import sys
from prospect.communication import Scheduler
from prospect.io import prepare_run, read_config
from prospect.input import Configuration, InputArgument

def run(user_inp: str) -> None:
    config_yaml = read_config(user_inp)
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
            state = prepare_run(user_inp, config)
            scheduler = Scheduler(config, state)
            scheduler.delegate(executor)
            scheduler.finalize()

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

    jobtype: jobtype
    mode: mode
    num_processes: num_processes