import pickle
import os
import sys 
import yaml
from mpi4py import MPI

if len(sys.argv) > 2:
    raise Exception('Invalid number of arguments. Only one is accepted.')

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if mpi_rank == 0:
    if os.path.isfile(sys.argv[1]):
        input_file = sys.argv[1]
        config = yaml.full_load(open(input_file, 'r'))
        print(f"Starting PROSPECT from input file {input_file}")
        state = {}
        if os.path.isdir(config['output_dir']):
            raise Exception('Your argument is already a folder. Remove it and try again!')
        os.system(f"mkdir {config['output_dir']}")
        os.system(f"cp {input_file} {config['output_dir']}")
    elif os.path.isdir(sys.argv[1]):
        output_folder = sys.argv[1]
        if not os.path.isfile(f"{output_folder}/state.pkl"):
            raise Exception('No state.pkl found in your argument folder. Please provide either a folder with a PROSPECT state.pkl dump or an input.yaml file.')
        else:
            with open(f"{output_folder}/state.pkl", "rb") as state_file:
                state = pickle.load(state_file)
            config = yaml.full_load(open(f"{output_folder}/config.yaml", 'r'))
        print(f"Resuming from PROSPECT snapshop in {output_folder}")
    else:
        raise Exception('Invalid arguments to PROSPECT. Give either a .yaml input file or the folder of a previous PROSPECT run.')
    if mpi_size > 1:
        print(f"Running PROSPECT on {mpi_size} processes...")
        from communication import Scheduler
        scheduler = Scheduler(config, state)
        scheduler.delegate()
        print("Scheduler finished")
    else:
        print("Running PROSPECT serially, without MPI...")
        from communication import Serial
        serial = Serial(config, state)
        serial.run()
else:
    from communication import Worker 
    worker = Worker()
    worker.work()
    print(f"Worker {mpi_rank} finished")

if mpi_rank == 0:
    print("PROSPECT finished. Enjoy!")

