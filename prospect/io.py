import os 
import pickle
import yaml 
import numpy as np 
from dataclasses import dataclass
from typing import Any
from prospect.input import InputArgument
from prospect.tasks.base_task import BaseTask

def read_config(arg: str) -> dict[str, Any]:
    if os.path.isfile(arg):
        config = yaml.full_load(open(arg, 'r'))
        config['io']['resume'] = False
    elif os.path.isdir(arg):
        # arg is folder, get the .yaml stored inside
        config = yaml.full_load(open(f"{arg}/log.yaml", 'r'))
        # Force the output directory to be the user input
        if config['io']['dir']:
            if arg != config['io']['dir']:
                print(f"The command line directory supplied is different from the output directory in the log.yaml file.")
                print(f"Changing the output directory to the command line directory.")
                config['io']['dir'] = arg
                config['io']['resume'] = True
    else:
        raise ValueError('Invalid arguments to PROSPECT. Give either a .yaml input file or the folder of a previous PROSPECT run.')
    return config

def prepare_run(arg: str, config):
    if not config.io.resume:
        BaseTask.idx_count = 1
        if config.io.write:
            os.makedirs(config.io.dir, exist_ok=False)
            print(f"Saving potentially modified config to {config.io.dir}/log.yaml")
            with open(os.path.join(config.io.dir, "log.yaml"), 'w') as outfile:
                yaml.dump(config.config_dict, outfile)
            print(f"Starting PROSPECT from input file {arg}.")
        return False # No state to resume from
    elif config.io.resume:
        # arg is folder, restart from that folder
        if not os.path.isfile(f"{arg}/state.pkl"):
            raise ValueError('No state.pkl found in your argument folder. Please provide either a folder with a PROSPECT state.pkl dump or an input.yaml file.')
        else:
            with open(f"{arg}/state.pkl", "rb") as state_file:
                state = pickle.load(state_file)
            max_id = max(set().union(*[state.ongoing, [task.id for task in state.queued], state.unready, state.done]))
            BaseTask.idx_count = max_id + 1
            print(f"Resuming PROSPECT snapshop in {arg}.")
        return state

def unpack_mcmc(param_dict, output_dir, jobname, *chain_list) -> None:
    """Unpacks the PROSPECT state into a usual cobaya-like MCMC folder."""
    write_parameters(param_dict, output_dir, jobname)
    write_chains(output_dir, jobname, *chain_list)

def write_parameters(param_dict, output_dir, jobname, latex_names=None) -> None:
    """Writes parameter names and ranges."""
    if latex_names == None:
        latex_names = list(param_dict.keys())

    with open(f"{output_dir}/{jobname}.paramnames", 'w') as file:
        for name, latex_name in zip(param_dict.keys(), latex_names):
            file.write(f"\n{name} {latex_name}")
    
    with open(f"{output_dir}/{jobname}.ranges", 'w') as file:
        for name, param_item in param_dict.items():
            file.write(f"\n{name} {str(param_item[0])} {str(param_item[1])}")

def write_chains(output_dir, jobname, *chain_list) -> None:
    """Stores chain contents to disk."""
    for idx_chain, chain in enumerate(chain_list):
        pos = np.array(chain.positions)
        output = np.append(np.stack([chain.mults, chain.loglkls], axis=1), pos, axis=1)
        np.savetxt(f"{output_dir}/{jobname}_{idx_chain}.txt",  output, delimiter=' ')


"""
    Definition of user arguments related to io.py

"""

@dataclass
class Arguments:
    class jobname(InputArgument):
        val_type = str
        default = None

    class write(InputArgument):
        val_type = bool
        default = False

    class dir(InputArgument):
        val_type = str
        default = None
        def validate(self, config):
            # Check that write = True
            pass
    
    class overwrite_dir(InputArgument):
        allowed_values = [True, False, 'yes', 'y']
        def get_default(self, config_yaml: dict[str, Any]):
            return False
        def validate(self, config):
            # Check that write = True
            pass
    
    class snapshot_interval(InputArgument):
        val_type = float
        default = 0.1 # Dumps snapshot whenever task finishes
        def validate(self, config):
            # Check that write = True
            pass
    
    class resume(InputArgument):
        # Whether the current run is the resume of a previous run
        val_type = bool
    
    jobname: jobname
    write: write
    dir: dir
    overwrite_dir: overwrite_dir
    snapshot_interval: snapshot_interval
    resume: resume