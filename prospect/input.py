import contextlib
import inspect
import os
from abc import ABC
from dataclasses import dataclass, field
from importlib import import_module
from types import UnionType
from typing import Any, get_args, get_origin, Union
import numpy as np 

class Configuration:
    def __init__(self, config_yaml):
        """config_yaml is the dictionary read from the input .yaml file."""
        print("Reading input.")
        # When new arguments are implemented, add them here
        modules = {
            'run':    import_module('prospect.run'), 
            'io':     import_module('prospect.io'),
            'kernel': import_module('prospect.kernels.base_kernel')
        }
        if 'run' in config_yaml and 'jobtype' in config_yaml['run']:
            if config_yaml['run']['jobtype'] == 'mcmc':
                modules['mcmc'] = import_module('prospect.mcmc')
            elif config_yaml['run']['jobtype'] == 'profile':
                modules['profile'] = import_module('prospect.profile')
            else:
                raise ValueError("The given value of 'jobtype' is not recognised. Choose either 'mcmc' or 'profile'.")
        else:
            raise ValueError("You must give 'jobtype' as an input under the 'run' category.")

        str_eval_arguments = [('profile', 'values')]
        for arg_module, arg_name in str_eval_arguments:
            if arg_module in config_yaml:
                if arg_name in config_yaml[arg_module]:
                    if type(config_yaml[arg_module][arg_name]) == str:
                        config_yaml[arg_module][arg_name] = safe_eval(config_yaml[arg_module][arg_name])

        self.set_defaults_iterative(modules, config_yaml)
        self.validate_parameters(modules, config_yaml)
        self.set_output_dir(config_yaml['io'])

        # Set parameters as attributes
        self.config_dict = config_yaml
        for module_name, module in modules.items():
            setattr(self, module_name, module.Arguments(**config_yaml[module_name]))
        
        print("Succeeded in setting defaults and validating inputs.")
        print("------------------------------")
    

    def set_defaults_iterative(self, modules: list, config_yaml) -> None:
        num_read = 1
        while num_read > 0:
            num_read = 0
            failed_args = []

            # Set default values
            for module_name, module in modules.items():
                print(f"Setting defaults for module '{module_name}'.")
                
                for arg_name, input_arg in self.arguments_iterator(module.Arguments):
                    try:
                        if not arg_name in config_yaml[module_name]:
                            config_yaml[module_name][arg_name] = input_arg().get_default(config_yaml)
                            print(f"Argument {arg_name} not found in input, set to default value {config_yaml[module_name][arg_name]}.")
                            num_read += 1
                    except Exception as e:
                        print(f"Failed setting parameter '{arg_name}'. Exception message:\n{e}")
                        failed_args.append(arg_name)
            if num_read == 0:
                if not failed_args:
                    print(f"End of iterative default setting, continuing to validation.")
                else:
                    raise ValueError(f"Could not set default value for inputs {failed_args}.")
            else:
                print(f"Set {num_read} default parameter(s), reiterating.")

    def arguments_iterator(self, arguments) -> list[str, Any]:
        # arguments is an instance of an Arguments class defined in the particular module
        out = inspect.getmembers(arguments, inspect.isclass)
        for idx, (name, arg) in enumerate(out):
            if name == '__class__':
                out.pop(idx)
        return out

    def validate_parameters(self, modules, config_yaml) -> None:
        for module_name, module in modules.items():
            print(f"Validating input from {module_name}.")
            for arg_name, input_arg in self.arguments_iterator(module.Arguments):
                self.validate_generic(input_arg, config_yaml[module_name][arg_name])
                input_arg().validate(config_yaml[module_name])

    def validate_generic(self, arg, arg_val: Any) -> None:
        if arg.val_type is not None:
            if get_origin(arg.val_type) in [Union, UnionType]:
                if not type(arg_val) in get_args(arg.val_type):
                    raise ValueError(f"Input '{arg().name}' was given type value of {type(arg_val)} which is not one of the allowed types: {get_args(arg.val_type)}.")
            else:
                if not type(arg_val) == arg.val_type:
                    raise ValueError(f"Input '{arg().name}' was given type value of {type(arg_val)}, but only type {arg.val_type} is allowed.")
        if arg.allowed_values is not None:
            if not arg_val in arg.allowed_values:
                raise ValueError(f"Input '{arg().name}' was given value {arg_val} which not in the list of allowed values: {arg.allowed_values}")

    def set_output_dir(self, config_io) -> None:
        # Sets output dir depending on whether to overwrite
        if config_io['write']:
            if not config_io['overwrite_dir']:
                if os.path.isdir(config_io['dir']):
                    output_idx = 0
                    while True:
                        if os.path.isdir(f"{config_io['dir']}_{output_idx}"):
                            output_idx += 1
                        else:
                            break
                    config_io['dir'] = f"{config_io['dir']}_{output_idx}"

def safe_eval(code):
    @contextlib.contextmanager
    def disable_imports():
        import builtins
        __import__ = builtins.__import__
        builtins.__import__ = None
        try:
            yield
        finally:
            builtins.__import__ = __import__
    scope = {'np': np, 'numpy': np}
    with disable_imports():
        try:
            return eval(code, scope)
        except Exception:
            raise ValueError(f'Illegal input {code!r}. Note that you may only use the standard and numpy library in Python-evaluted statements in the .yaml file.')

@dataclass
class InputArgument(ABC):
    # If set, will check that the correct type is given
    val_type: Any = field(default=None) 

    # If not specified, all values are allowed
    allowed_values: list = field(default=None)

    # Specify default values as functions of other arguments
    def get_default(self, config_yaml: dict[str, Any]):
        # config_yaml is the entire input dictionary
        raise ValueError(f"The input '{self.name}' has no default and must be specified.")
        
    def validate(self, config: dict[str, Any]) -> None:
        # config is the config_yaml[module] input dict, after settings default values, where module is the module that this argument is defined in 
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
