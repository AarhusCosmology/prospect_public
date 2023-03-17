import os
import pickle
import shutil
import yaml
from typing import Any
from prospect.communication import TasksState

def read_config(arg: str) -> dict[str, Any]:
    if os.path.isfile(arg):
        config = yaml.full_load(open(arg, 'r'))
    elif os.path.isdir(arg):
        # arg is folder, get the .yaml stored inside
        config = yaml.full_load(open(f"{arg}/{arg}.yaml", 'r'))
    else:
        raise ValueError('Invalid arguments to PROSPECT. Give either a .yaml input file or the folder of a previous PROSPECT run.')
    check_requirements(config)
    check_valid_values(config)
    set_defaults(config)
    set_output_dir(config)
    return config

def check_requirements(config: dict[str, Any]) -> None:
    pass

def check_valid_values(config: dict[str, Any]) -> None:
    pass

def set_defaults(config: dict[str, Any]) -> None:
    if 'n_procs' not in config:
        if config['run_mode'] == 'mpi':
            from mpi4py import MPI
            config['n_procs'] = MPI.COMM_WORLD.Get_size()
        elif config['run_mode'] == 'threaded':
            config['n_procs'] = os.cpu_count() or 1
        elif config['run_mode'] == 'serial':
            config['n_procs'] = 1
    if 'overwrite_output' not in config:
        config['overwrite_output'] = False

def set_output_dir(config: dict[str, Any]) -> None:
    # Sets output dir depending on whether to overwrite
    if config['write_output']:
        if os.path.isdir(config['output_dir']):
            if config['overwrite_output']:
                shutil.rmtree(config['output_dir'])
            else:
                output_idx = 0
                while True:
                    if os.path.isdir(f"{config['output_dir']}_{output_idx}"):
                        output_idx += 1
                    else:
                        break
                config['output_dir'] = f"{config['output_dir']}_{output_idx}"
                    
    print(config['output_dir'])

def prepare_run(arg: str, config: dict[str, Any]) -> bool | TasksState:
    if os.path.isfile(arg):
        # arg is an input file, start from it
        if config['write_output']:
            os.makedirs(config['output_dir'], exist_ok=False)
            shutil.copy(arg, config['output_dir'])
            print(f"Starting PROSPECT from input file {arg}.")
        return False # No state to resume from
    elif os.path.isdir(arg):
        # arg is folder, restart from that folder
        if not os.path.isfile(f"{arg}/state.pkl"):
            raise ValueError('No state.pkl found in your argument folder. Please provide either a folder with a PROSPECT state.pkl dump or an input.yaml file.')
        else:
            with open(f"{arg}/state.pkl", "rb") as state_file:
                state = pickle.load(state_file)
            print(f"Resuming from PROSPECT snapshop in {arg}.")
        return state