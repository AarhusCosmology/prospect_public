import sys
from prospect.communication import Scheduler
from prospect.input import read_config, prepare_run

def run(arg: str) -> None:
    config = read_config(arg)

    if config['run_mode'] == 'mpi':
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            raise ValueError('You cannot run PROSPECT using MPI with only one process.')
        pool = MPICommExecutor
        pool_args, pool_kwargs = [comm], {'root': 0}
    elif config['run_mode'] == 'threaded':
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor
        pool_args, pool_kwargs = [], {'max_workers': config['n_procs']}
    elif config['run_mode'] == 'serial':
        from prospect.communication import SerialContext
        pool = SerialContext
        pool_args, pool_kwargs = [], {}
    else:
        raise ValueError(f"Run mode '{config['run_mode']}' not recognized. Choose either 'mpi', 'threaded' or 'serial'.")

    with pool(*pool_args, **pool_kwargs) as executor:
        if executor is not None:
            print(f"Running PROSPECT with mode *{config['run_mode']}* on {config['n_procs']} processes...")
            state = prepare_run(arg, config)
            scheduler = Scheduler(config, state)
            scheduler.delegate(executor)

    if config['run_mode'] == 'mpi':
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
