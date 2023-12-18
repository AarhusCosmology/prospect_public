from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import numpy as np 
from typing import Any, Type
from types import NoneType
from prospect.input import InputArgument

class BaseKernel(ABC):
    def __init__(self, config_kernel, task_id, output_folder=None):
        self.set_default_errors()
        self.id = task_id
        self.initialise(config_kernel, output_folder)

        self.param = {
            'varying': {},
            'fixed': {},
            'derived': {}
        }
        self.set_parameter_dict()
        self.save_config()
    
    class NullException(Exception):
        pass

    def set_default_errors(self):
        """
            Two error types exist: 
                Severe exceptions will stop the process
                Computation exceptions will assign -inf as the 
                likelihood value of the proposed point and then continue the run

            Derivatives of BaseKernel can define their own specific error types.
        """
        # By default, all exceptions are severe
        self.severe_exception = Exception 
        self.computation_exception = self.NullException

    @abstractmethod
    def initialise(self, kernel_param):
        pass

    def set_fixed_parameters(self, fixed_param_dict):
        for param_name, fixed_value in fixed_param_dict.items():
            # Move parameter to the fixed dict
            self.param['fixed'][param_name] = self.param['varying'][param_name]
            self.param['fixed'][param_name]['fixed_value'] = fixed_value
            del self.param['varying'][param_name]
    
    @property
    def varying_param_names(self):
        return list(self.param['varying'].keys())
    
    @abstractmethod
    def set_parameter_dict(self):
        """
            Must set self.param, a dict with keys that are parameter names
        """
        pass

    def save_config(self):
        """
        Saves kernel-specific data in the kernel subdir based on type
            - Analytical: Nothing saved
            - MontePython: .conf and .param
        """
        pass

    def loglkl(self, prop):
        for fixed_param, param in self.param['fixed'].items():
            prop[fixed_param] = [param['fixed_value']]
        try:
            return self._loglkl(prop)
        except self.computation_exception as e:
            print(f"Soft exception {e} occurred. Trying a new proposal.")
            return np.inf
        except self.severe_exception:
            raise ValueError("Severe exception occurred in likelihood computation. Stopping process.")
    
    def log_uniform_prior(self, position):
        if self.outside_of_prior_bound(position):
            return np.inf
        # Note: Different normalization than cobaya, same as MontePython
        return 0.0
    
    def outside_of_prior_bound(self, prop):
        outside = False
        for param_name, param_val in prop.items():
            if param_name in self.param['varying']:
                prior = self.param['varying'][param_name]['range']
                if prior[0] is not None:
                    if param_val[0] < prior[0]:
                        outside = True
                        break
                if prior[1] is not None:
                    if param_val[0] > prior[1]:
                        outside = True
                        break
        return outside

    @abstractmethod
    def _loglkl(self, position):
        # Should return -log(likelihood); note the minus sign!
        pass

    @abstractmethod
    def logprior(self, position):
        # Should return -log(prior); note the minus sign!
        pass

    def logpost(self, position):
        return self.loglkl(position)*self.logprior(position)

    @abstractmethod
    def get_default_initial_position(self):
        # Return default initial position
        pass

    @abstractmethod
    def read_initial_position(self, config_initial_position):
        # Return initial position given the user input
        pass

    @abstractmethod
    def get_default_covmat(self):
        # Return default (initial) covmat
        pass

    @abstractmethod
    def read_covmat(self, config_covmat):
        # Return (initial) covmat given user input
        pass

    def finalize(self):
        pass

"""
    Definition of user arguments related to base_kernel.py
"""

@dataclass
class Arguments:
    class type(InputArgument):
        val_type = str
        allowed_values = ['analytical', 'montepython', 'cobaya']
    
    class param(InputArgument):
        val_type = str # either path to .param/.yaml-file or dict if analytical type
        def validate(self, config: dict[str, Any]) -> None:
            if not os.path.isfile(os.path.join(os.getcwd(), config['param'])):
                raise ValueError(f"The file pointed to in the 'param' field of the 'kernel' input, which has the value {config['param']}, could not be found.")
    
    class conf(InputArgument):
        val_type = str
        def get_default(self, config_yaml: dict[str, Any]):
            if config_yaml['kernel']['type'] == 'analytical' or config_yaml['kernel']['type'] == 'cobaya':
                return ''
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['type'] == 'montepython':
                if not os.path.isfile(os.path.join(os.getcwd(), config['param'])):
                    raise ValueError(f"The file pointed to in the 'conf' field of the 'kernel' input, which has the value {config['conf']}, could not be found.")
    
    class path(InputArgument):
        # Only use if MontePython: Should point to the 'montepython' directory
        val_type = str
        def get_default(self, config_yaml: dict[str, Any]):
            if config_yaml['kernel']['type'] == 'analytical' or config_yaml['kernel']['type'] == 'cobaya':
                return ''
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['type'] == 'montepython':
                if not os.path.isdir(os.path.join(os.getcwd(), config['path'])):
                    raise ValueError(f"The directory pointed to in the 'path' field of the 'kernel' input, which has the value {config['path']}, could not be found.")

    class debug(InputArgument):
        # For cobaya: Debug mode, drops debug files in kernel subfolder
        val_type = bool
        def get_default(self, config_yaml: dict[str, Any]):
            return False

    type: type
    param: param
    conf: conf
    path: path
    debug: debug
