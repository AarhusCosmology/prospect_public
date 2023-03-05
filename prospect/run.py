import os
import pickle
import yaml
from mpi4py import MPI

def run(input: str):
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    if mpi_rank == 0:
        print(input, type(input))
        if os.path.isfile(input):
            config = yaml.full_load(open(input, 'r'))
            print(f"Starting PROSPECT from input file {input}")
            state = {}
            if os.path.isdir(config['output_dir']):
                raise ValueError('Your argument is already a folder. Remove it and try again!')
            os.system(f"mkdir {config['output_dir']}")
            os.system(f"cp {input} {config['output_dir']}")
        elif os.path.isdir(input):
            if not os.path.isfile(f"{input}/state.pkl"):
                raise ValueError('No state.pkl found in your argument folder. Please provide either a folder with a PROSPECT state.pkl dump or an input.yaml file.')
            else:
                with open(f"{input}/state.pkl", "rb") as state_file:
                    state = pickle.load(state_file)
                config = yaml.full_load(open(f"{input}/config.yaml", 'r'))
            print(f"Resuming from PROSPECT snapshop in {input}")
        else:
            raise ValueError('Invalid arguments to PROSPECT. Give either a .yaml input file or the folder of a previous PROSPECT run.')
        if mpi_size > 1:
            print(f"Running PROSPECT on {mpi_size} processes...")
            from prospect.communication import Scheduler
            scheduler = Scheduler(config, state)
            scheduler.delegate()
            print("Scheduler finished")
        else:
            print("Running PROSPECT serially, without MPI...")
            from prospect.communication import Serial
            serial = Serial(config, state)
            serial.run()
    else:
        from prospect.communication import Worker
        worker = Worker()
        worker.work()
        print(f"Worker {mpi_rank} finished")

    if mpi_rank == 0:
        print("PROSPECT finished. Enjoy!")

def run_from_shell():
    """
        Wrapper for run() using a setuptools entry point
        Allows running from command-line with the command `prospect input/test.yaml`
        Only takes arguments using argparse

    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='+')
    args = parser.parse_args()
    print(f"Running PROSPECT from shell with input {args.input_file[0]}")
    run(args.input_file[0])