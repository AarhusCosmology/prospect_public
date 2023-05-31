from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import numpy as np 
from typing import Any
from prospect.input import InputArgument

class BaseKernel(ABC):
    def __init__(self, config_kernel, task_id, output_folder=None):
        self.id = task_id
        self.initialise(config_kernel, output_folder)
        self.set_parameter_dict()

    @abstractmethod
    def initialise(self, kernel_param):
        pass

    @abstractmethod
    def set_parameter_dict(self):
        """
            Must set self.param, a dict with keys that are parameter names
            Values must be a dict with keys:
                - prior: a tuple of (lower, upper) bound of the uniform prior
                
        """
        pass

    @abstractmethod
    def loglkl(self, position):
        # Should return -log(likelihood); note the minus sign!
        pass

    @abstractmethod
    def get_initial_position(self, config_initial_position):
        # Return initial position given the user input
        pass

    @abstractmethod
    def get_default_initial_position(self):
        # Give a good guess of the initial position
        pass

    @abstractmethod
    def get_covmat(self, config_covmat):
        # Return (initial) covmat given user input
        pass

    @abstractmethod
    def get_default_covmat(self):
        # Get a default parameter covmat
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
            assert os.path.isfile(os.path.join(os.getcwd(), config['param']))
    
    class conf(InputArgument):
        val_type = str
        def get_default(self, config_yaml: dict[str, Any]):
            if config_yaml['kernel']['type'] == 'analytical':
                return ''
            return None

        def validate(self, config: dict[str, Any]) -> None:
            assert os.path.isfile(os.path.join(os.getcwd(), config['param']))
    
    type: type
    param: param
    conf: conf
