from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
import os 
import time
from typing import Type
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.interpolate import interp1d
from scipy import stats
from prospect.input import Configuration
from prospect.kernels.initialisation import initialise_kernel
from prospect.tasks.base_task import BaseTask
from prospect.tasks.optimise_task import OptimiseTask

# Switch matplotlib backend to one that doesn't pop up figures
matplotlib.use('Agg')

def argmax(iterable):
    """From https://stackoverflow.com/questions/16945518/finding-the-index-of-the-value-which-is-the-min-or-max-in-python"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]

class BaseAnalyseTask(BaseTask, ABC):
    priority = 75.0

    # Figure settings
    cm = 1/2.54  # centimeters in inches
    fig = {
        'width': 8.6*cm, # PRL figure width
        'height': 2,
        'fontsize': 11/1.1,
        'rep_colors': ['b', 'r', 'g', 'c', 'y', 'm']*50,
        'rep_ms': [4.5, 4.0, 3.5, 3, 2.5, 2., 1.5, 1., 0.5]*50,
        'interval_styles': ['--', ':']*5
    }

    def __init__(self, config: Configuration, required_task_ids: list[int]):
        super().__init__(config, required_task_ids)
        self.dir = os.path.join(config.io.dir, config.run.jobtype)
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)


    @abstractmethod
    def run(self, tasks: list[OptimiseTask]): 
        pass

    def emit_tasks(self) -> list[Type[BaseTask]]:
        # Never emit anything; the OptimiseTasks manage convergence themselves.
        return []
