import argparse
import yaml
from mpi4py import MPI

parser = argparse.ArgumentParser(description='')
parser.add_argument('-p', metavar='CONFIG', type=str, help="PROSPECT configuration file.")
args = parser.parse_args()
config = yaml.full_load(open(args.p, 'r'))

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if mpi_size == 1:
    print(f"Running PROSPECT serially, without MPI...")
    from communication import Serial
    serial = Serial(config)
    serial.run()
else:
    if mpi_rank == 0:
        print(f"Running PROSPECT on {mpi_size} processes...")
        from communication import Scheduler
        scheduler = Scheduler(config)
        scheduler.delegate()
    elif mpi_rank > 0:
        from communication import Worker 
        worker = Worker()
        worker.work()

# Call analysis.py and write output (or implement this as new tasks)

if mpi_rank == 0:
    print(f"PROSPECT finished. Enjoy!")

