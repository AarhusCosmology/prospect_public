import os 
import pickle
import shutil
import yaml 
import numpy as np 
from dataclasses import dataclass
from typing import Any
from prospect.input import InputArgument
from prospect.tasks.base_task import BaseTask

def read_user_input(arg: str) -> dict[str, Any]:
    if os.path.isfile(arg):
        resume = False
        config = yaml.full_load(open(arg, 'r'))
    elif os.path.isdir(arg):
        resume = True
        # arg is folder, get the .yaml stored inside
        config = yaml.full_load(open(f"{arg}/log.yaml", 'r'))
        # Force the output directory to be the user input
        if config['io']['dir']:
            if arg != config['io']['dir']:
                print(f"The command line directory supplied is different from the output directory in the log.yaml file.")
                print(f"Changing the output directory to the command line directory.")
                config['io']['dir'] = arg
    else:
        raise ValueError('Invalid arguments to PROSPECT. Give either a .yaml input file or the folder of a previous PROSPECT run.')
    return config, resume

def load_config(arg: str):
    # arg is folder, restart from that folder
    if not os.path.isfile(f"{arg}/config.pkl"):
        raise ValueError(f'Could not find {arg}/config.pkl.')
    else:
        with open(f"{arg}/config.pkl", "rb") as config_file:
            config = pickle.load(config_file)
        return config

def load_state(arg: str):
    # arg is folder, restart from that folder
    if not os.path.isfile(f"{arg}/state.pkl"):
        raise ValueError('No state.pkl found in your argument folder. Please provide either a folder with a PROSPECT state.pkl dump or an input.yaml file.')
    else:
        with open(f"{arg}/state.pkl", "rb") as state_file:
            state = pickle.load(state_file)
        max_id = max(set().union(*[state.ongoing, [task.id for task in state.queued], state.unready, state.done]))
        BaseTask.idx_count = max_id + 1
        return state

def prepare_run(config):
    BaseTask.idx_count = 1
    if config.io.write:
        # Setup job directory
        if os.path.isdir(config.io.dir):
            if config.io.overwrite_dir:
                shutil.rmtree(config.io.dir)
        os.makedirs(config.io.dir, exist_ok=False)
        # Create kernel directory
        os.makedirs(os.path.join(config.io.dir, config.run.jobtype))
        if config.kernel.type in ['montepython', 'cobaya', 'analytical']:
            os.makedirs(os.path.join(config.io.dir, config.kernel.type))
            # If MP: Save MP stuff
            # If cobaya: Save cobaya stuff
        # Save log.yaml
        print(f"Saving potentially modified config to {config.io.dir}/log.yaml")
        for module_name, module in config.config_dict.items():
            for param_name, param_value in module.items():
                # yaml cannot save numpy arrays, so convert them to lists before storing
                if isinstance(param_value, np.ndarray):
                    config.config_dict[module_name][param_name] = param_value.tolist()
        with open(os.path.join(config.io.dir, "log.yaml"), 'w') as outfile:
            yaml.dump(config.config_dict, outfile)

def unpack_mcmc(param_dict, output_dir, jobname, *chain_list) -> None:
    """Unpacks the PROSPECT state into a usual cobaya-like MCMC folder."""
    write_parameters(param_dict, output_dir, jobname)
    write_chains(output_dir, jobname, *chain_list)

def find(s, ch):
    """Utility for finding indices of string where character matches ch
       from https://stackoverflow.com/questions/11122291/how-to-find-char-in-string-and-get-all-the-indexes"""
    return [i for i, ltr in enumerate(s) if ltr == ch]

def remove_double_subscripts_latex(latex_names):
    """Reformats all double subscripts in latex names
       e.g. changes Omega_A_B_C to Omega_{A,B,C}"""
    for idx, name in enumerate(latex_names):
        underscore_indices = find(name, '_')
        if len(underscore_indices) > 1:
            # Double subscript
            latex_names[idx] = name[0:underscore_indices[0]] + '{' + name[underscore_indices[0]:].replace('_', ',') + '}'
    return latex_names

def write_parameters(param_dict, output_dir, jobname, latex_names=None) -> None:
    """Writes parameter names and ranges."""

    # If a name contains *, replace it by _ so as to be understood by GetDist
    reformatted_names = {}
    for name in param_dict.keys():
        new_name = name
        # Put all name replacements here
        if '*' in name:
            new_name = name.replace('*', '_')
        reformatted_names[name] = new_name
    # Doing it this way preserves the old ordering of the dict 
    param_dict = {reformatted_names[name]: param_dict[name] for name in param_dict.keys()}

    if latex_names == None:
        latex_names = remove_double_subscripts_latex(list(param_dict.keys()))

    with open(f"{output_dir}/{jobname}.paramnames", 'w') as file:
        for name, latex_name in zip(param_dict.keys(), latex_names):
            file.write(f"\n{name} {latex_name}")
    
    with open(f"{output_dir}/{jobname}.ranges", 'w') as file:
        for name, param_item in param_dict.items():
            write_str = f"\n{name}"
            for bound in param_item['range']:
                if bound is None:
                    write_str += " N"
                else:
                    write_str += f" {str(bound)}"
            file.write(write_str)

def write_chains(output_dir, jobname, *chain_list) -> None:
    """Stores chain contents to disk."""
    for idx_chain, chain in enumerate(chain_list):
        np.savetxt(f"{output_dir}/{jobname}_{idx_chain}.txt",  chain.data, delimiter=' ')


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
        # Dumps snapshot whenever task finishes
        def get_default(self, config_yaml: dict[str, Any]):
            return 1.0
        def validate(self, config):
            # Check that write = True
            pass
    
    jobname: jobname
    write: write
    dir: dir
    overwrite_dir: overwrite_dir
    snapshot_interval: snapshot_interval