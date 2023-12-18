import ast
import contextlib
import copy
import os
import shutil
import sys
from typing import Any
import numpy as np 
from prospect.kernels.base_kernel import BaseKernel

class InitialiseMontePython:
    """
        Singleton class that initialises MP on first init and returns
        that instance at every other initialisation, meaning
        MP is only ever initialised once
        (thus does not support MP inits with different input...)
    """
    def __new__(cls, config_kernel, output_folder, id):
        if not hasattr(cls, 'mp'):
            print("Initialising MontePython on process for the first time.")
            with contextlib.redirect_stdout(open(os.path.join(f"{output_folder}/montepython", f'id_{id}.out'), 'w')):
                with contextlib.redirect_stderr(open(os.path.join(f"{output_folder}/montepython", f'id_{id}.err'), 'w')):
                    cls.mp = cls.initialise_montepython(config_kernel, output_folder, id)
            #cls.mp = cls.initialise_montepython(config_kernel, output_folder, id)
        else:
            print("MontePython was already initialised, returning old instance.")
            if config_kernel.param != cls.mp['param']:
                raise ValueError(f"Tried to initialise MontePython with .param file {config_kernel.param}, which is different to {cls.mp['param']}, which it was firstly initialised with.")
            elif config_kernel.conf != cls.mp['conf']:
                raise ValueError(f"Tried to initialise MontePython with .conf file {config_kernel.conf}, which is different to {cls.mp['conf']}, which it was firstly initialised with.")
        return cls.mp
    
    def __reduce__(self) -> str | tuple[Any, ...]:
        return (None, None)

    def initialise_montepython(config_kernel, output_folder, id):
        mp = {}
        sys.path.append(f"{config_kernel.path}")
        try:
            from initialise import initialise as mp_initialise
            import sampler
            mp['compute_lkl'] = sampler.compute_lkl
            mp['get_covmat'] = sampler.get_covariance_matrix
            mp['read_args'] = sampler.read_args_from_bestfit
            from io_mp import CosmologicalModuleError
            mp['cosmo_soft_exception'] = CosmologicalModuleError
        except Exception:
            raise ImportError(f'Could not import MontePython modules. Is the path you gave, {config_kernel.path}, correctly pointing to the /montepython_public/montepython directory?')

        mp_dir = f"{output_folder}/montepython/id_{id}"
        mp_command_input = f'run -p {config_kernel.param} --conf {config_kernel.conf} -o {mp_dir} --chain-number 0'
        mp['param'], mp['conf'] = config_kernel.param, config_kernel.conf
        mp['cosmo'], mp['data'], mp['command_line'], _ = mp_initialise(mp_command_input)
        mp['err_dir'], mp['out_dir'] = f"{output_folder}/montepython/id_{id}.err", f"{output_folder}/montepython/id_{id}.out"
        return mp 

class MontePythonKernel(BaseKernel):
    def initialise(self, config_kernel, output_folder):
        self.mp = InitialiseMontePython(config_kernel, output_folder, self.id)
        self.dimension = len(list(self.mp['data'].mcmc_parameters.keys()))
        self.output_folder = output_folder
        self.config_kernel = config_kernel

        from classy import CosmoSevereError
        self.severe_exception = CosmoSevereError
        self.computation_exception = self.mp['cosmo_soft_exception']

    def set_parameter_dict(self):
        self.mp_to_prospect_name = {}
        for param_name, param_dict in self.varying_param_dict.items():
            prospect_param_name = self.format_param_name(param_name)
            self.param['varying'][prospect_param_name] = {'range': param_dict['prior'].prior_range}
            self.mp_to_prospect_name[param_name] = prospect_param_name
        for param_name in self.get_mp_param_names('derived'):
            prospect_param_name = self.format_param_name(param_name)
            self.param['derived'][prospect_param_name] = {}
            self.mp_to_prospect_name[param_name] = prospect_param_name
        self.prospect_to_mp_name = {val: key for key, val in self.mp_to_prospect_name.items()}

    def format_param_name(self, param_name):
        # Translate a parameter name from MontePython format to PROSPECT format
        prospect_param = param_name
        if '*' in param_name:
            prospect_param = param_name.replace('*', '')
        return prospect_param
    
    def save_config(self):
        shutil.copy(self.config_kernel.conf, os.path.join(self.output_folder, 'montepython/log.conf'))
        shutil.copy(self.config_kernel.param, os.path.join(self.output_folder, 'montepython/log.param'))

    def _loglkl(self, position):
        for param_name, param in self.param['varying'].items():
            self.mp['data'].mcmc_parameters[self.prospect_to_mp_name[param_name]]['current'] = position[param_name][0]
        for param_name, param in self.param['fixed'].items():
            self.mp['data'].mcmc_parameters[self.prospect_to_mp_name[param_name]]['current'] = param['fixed_value']
        self.mp['data'].update_cosmo_arguments()
        with contextlib.redirect_stdout(open(self.mp['out_dir'], 'a+')):
            with contextlib.redirect_stderr(open(self.mp['err_dir'], 'a+')):
                out = self.mp['compute_lkl'](self.mp['cosmo'], self.mp['data'])
        return -out

    def logprior(self, position):
        return self.log_uniform_prior(position)

    def read_initial_position(self, config_initial_position):
        # Reads a MontePython .bestfit file
        # If a parameter is not given in the .bestfit file, default value
        # for that parameter will be set
        with contextlib.redirect_stdout(open(self.mp['out_dir'], 'a+')):
            with contextlib.redirect_stderr(open(self.mp['err_dir'], 'a+')):
                self.mp['read_args'](self.mp['data'], config_initial_position)
        initial = {}
        for param_name, param in self.varying_param_dict.items():
            if 'last_accepted' in param:
                initial[self.mp_to_prospect_name[param_name]] = [param['last_accepted']]
            else:
                initial[self.mp_to_prospect_name[param_name]] = [param['initial'][0]]
        return initial

    def get_default_initial_position(self):
        # Default: Set initial values to mean values given in MP parameter file
        initial = {}
        for param_name, param in self.varying_param_dict.items():
            initial[param_name] = [param['initial'][0]]
        return initial

    def read_covmat(self, config_covmat):
        self.mp['command_line'].cov = config_covmat
        with contextlib.redirect_stdout(open(self.mp['out_dir'], 'a+')):
            with contextlib.redirect_stderr(open(self.mp['err_dir'], 'a+')):
                eigval, eigvec, covmat = self.mp['get_covmat'](self.mp['cosmo'], self.mp['data'], self.mp['command_line'])
        return covmat

    def get_default_covmat(self):
        # Extract default from MontePython
        with contextlib.redirect_stdout(open(self.mp['out_dir'], 'a+')):
            with contextlib.redirect_stderr(open(self.mp['err_dir'], 'a+')):
                eigval, eigvec, covmat = self.mp['get_covmat'](self.mp['cosmo'], self.mp['data'], self.mp['command_line'])
        return covmat

    def get_mp_param_names(self, type):
        return self.mp['data'].get_mcmc_parameters([type])

    @property
    def varying_param_dict(self):
        return {param_name: self.mp['data'].mcmc_parameters[param_name] for param_name in self.mp['data'].get_mcmc_parameters(['varying'])}
