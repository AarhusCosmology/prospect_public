import os
from typing import Any
import yaml
import numpy as np 
from prospect.input import safe_eval
from prospect.kernels.base_kernel import BaseKernel

class InitialiseCobaya:
    """
        Singleton-type class that initialises cobaya on first init and returns
        that instance at every other initialisation, meaning
        cobaya is only ever initialised once
        (thus does not support inits with different input...)
    """

    def __new__(cls, config_kernel, output_folder, id):
        if not hasattr(cls, 'model'):
            print("Initialising cobaya on process for the first time.")
            cls.model, cls.sampler, cls.yaml = cls.initialise_cobaya(config_kernel, output_folder, id)
            cls.param = config_kernel.param
        else:
            print("cobaya was already initialised, returning old instance.")
            if config_kernel.param != cls.param:
                raise ValueError(f"Tried to initialise MontePython with .param file {config_kernel.param}, which is different to {cls.param}, which it was firstly initialised with.")
        return cls.model, cls.sampler, cls.yaml
    
    def __reduce__(self) -> str | tuple[Any, ...]:
        return (None, None)
    
    def initialise_cobaya(config_kernel, output_folder, id):
        try:
            os.environ['COBAYA_NOMPI'] = "False"
            from cobaya.model import get_model
            from cobaya.yaml import yaml_load_file
            from cobaya.sampler import get_sampler
            extra_cobaya_settings = {
                'output': os.path.join(output_folder, f'cobaya/{id}/{id}')
            }
            if config_kernel.debug:
                extra_cobaya_settings['debug'] = os.path.join(output_folder, f'cobaya/{id}.debug')
            else:
                extra_cobaya_settings['debug'] = 40

            cobaya_yaml = yaml_load_file(config_kernel.param)
            cobaya_yaml.update(extra_cobaya_settings)
            cobaya_model = get_model(cobaya_yaml)
            cobaya_sampler = get_sampler({'mcmc': ''}, cobaya_model)
            return cobaya_model, cobaya_sampler, cobaya_yaml
        except ImportError:
            raise ImportError('Could not import cobaya. Make sure you have cobaya installed!')

class CobayaKernel(BaseKernel):
    def initialise(self, config_kernel, output_folder):
        self.cobaya_model, self.cobaya_sampler, self.cobaya_yaml = InitialiseCobaya(config_kernel, output_folder, self.id)
        self.dir = os.path.join(output_folder, 'cobaya')
        self.save_config()

        """
        if 'theory' in self.cobaya_yaml:
            if 'class' in self.cobaya_yaml:
                from classy import CosmoSevereError, CosmoComputationError
                self.severe_exception = CosmoSevereError
                self.computation_exception = CosmoComputationError
                # In camb, all errors are severe exceptions
        """

    def _loglkl(self, position):
        # Have to write it like this to unpack the value lists of 'positions'
        return -self.cobaya_model.loglike({param_name: pos[0] for param_name, pos in position.items()})[0]
    
    def logprior(self, position):
        return -self.cobaya_model.logprior({param_name: pos[0] for param_name, pos in position.items()})

    def set_parameter_dict(self):
        for param_name, param_dict in self.cobaya_model.parameterization.sampled_params_info().items():
            self.param['varying'][param_name] = {'prior': param_dict['prior']}
            if 'min' in param_dict['prior'] and 'max' in param_dict['prior']:
                self.param['varying'][param_name]['range'] = [param_dict['prior']['min'], param_dict['prior']['max']]
            else:
                self.param['varying'][param_name]['range'] = [None, None]
        for param_name in self.cobaya_model.parameterization.derived_params():
            self.param['derived'][param_name] = {}

    def save_config(self):
        with open(os.path.join(self.dir, "log.yaml"), 'w') as log_yaml:
            yaml.dump(self.cobaya_yaml, log_yaml)

    def read_initial_position(self, config_initial_position):
        # Must point to the my_cobaya_run_minize.bestfit or my_cobaya_run_minimize.bestfit.txt
        if not config_initial_position.endswith('.txt'):
            config_initial_position += '.txt'
        initial = np.genfromtxt(config_initial_position, names=True)
        initial_data = np.array(safe_eval(str(initial)))
        initial_position = {}
        for idx, value in enumerate(initial_data):
            name = initial.dtype.names[idx]
            if name in self.param['varying']:
                initial_position[name] = [value]
        return initial_position
    
    def get_default_initial_position(self):
        return {param: [val] for param, val in zip(self.cobaya_sampler.current_point.sampled_params, self.cobaya_sampler.current_point.values)}

    def read_covmat(self, config_covmat):
        # Imitates CovmatSampler._load_covmat from cobaya source code
        return np.atleast_2d(np.loadtxt(config_covmat))

    def get_default_covmat(self):
        return self.cobaya_sampler.initial_proposal_covmat()[0]
