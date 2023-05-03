from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any
from prospect.input import InputArgument

class BaseKernel(ABC):
    def __init__(self, kernel_param):
        self.initialise(kernel_param)
        pass

    @abstractmethod
    def initialise(self, kernel_param):
        pass

    @abstractmethod
    def loglkl(self, position):
        # Should return -log(likelihood); note the minus sign!
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
            # How to catch the right path?
            assert os.path.isfile(os.path.join(os.getcwd(), config['param']))
    
    type: type
    param: param