import ast
import contextlib
import os
import sys
import numpy as np 
from prospect.kernels.base_kernel import BaseKernel

class MontePythonKernel(BaseKernel):
    def initialise(self, config_kernel, output_folder):
        mp_path = None
        print(f"Reading MontePython location from configuration file {config_kernel.conf}")
        for line in open(config_kernel.conf, 'r'):
            if line.startswith("root"):
                mp_path = ast.literal_eval(line.split('=')[1])
        if mp_path is None:
            raise ValueError('You must specify the path to the superdirectory of montepython_public folder as "root" in the .conf file.')
        
        sys.path.append(f"{mp_path}/montepython_public/montepython")
        try:
            from initialise import initialise as mp_initialise
            import sampler
            self.compute_lkl = sampler.compute_lkl
            self.mp_get_covmat = sampler.get_covariance_matrix
            self.mp_read_args = sampler.read_args_from_bestfit
        except:
            raise ImportError('Could not import MontePython modules. Did you specify the correct root path in the .conf file?')
        
        self.mp_dir = f"{output_folder}/montepython/id_{self.id}/"
        os.makedirs(self.mp_dir)
        mp_command_input = f'run -p {config_kernel.param} --conf {config_kernel.conf} -o {self.mp_dir} --chain-number {self.id}'
        mp = {}
        with contextlib.redirect_stdout(open(os.path.join(self.mp_dir, '.out'), 'w')):
            with contextlib.redirect_stderr(open(os.path.join(self.mp_dir, '.err'), 'w')):
                mp['cosmo'], mp['data'], mp['command_line'], _ = mp_initialise(mp_command_input)
        
        self.dimension = len(list(mp['data'].mcmc_parameters.keys()))
        self.mp = mp
        self.config_kernel = config_kernel
    
    def set_parameter_dict(self):
        self.param = {}
        for param_name, param_dict in self.varying_param_dict.items():
            self.param[param_name] = {}
            self.param[param_name]['prior'] = param_dict['prior'].prior_range

    def loglkl(self, position):
        for param_name, param_dict in self.varying_param_dict.items():
            param_dict['current'] = position[param_name]
        self.mp['data'].update_cosmo_arguments()
        with contextlib.redirect_stdout(open(os.path.join(self.mp_dir, '.out'), 'a+')):
            with contextlib.redirect_stderr(open(os.path.join(self.mp_dir, '.err'), 'a+')):
                out = self.compute_lkl(self.mp['cosmo'], self.mp['data'])
        return -out

    def get_initial_position(self, config_initial_position):
        # Reads a MontePython .bestfit file
        # If a parameter is not given in the .bestfit file, default value
        # for that parameter will be set
        with contextlib.redirect_stdout(open(os.path.join(self.mp_dir, '.out'), 'w')):
            with contextlib.redirect_stderr(open(os.path.join(self.mp_dir, '.err'), 'w')):
                self.mp_read_args(self.mp['data'], config_initial_position)
        initial = {}
        for param_name, param in self.varying_param_dict.items():
            if 'last_accepted' in param:
                initial[param_name] = param['last_accepted']
            else:
                initial[param_name] = param['initial'][0]
        return initial

    def get_default_initial_position(self):
        # Set initial values to mean values given in MP parameter file
        initial = {}
        for param_name, param in self.varying_param_dict.items():
            initial[param_name] = param['initial'][0]
        return np.array(initial)

    def get_default_covmat(self):
        with contextlib.redirect_stdout(open(os.path.join(self.mp_dir, '.out'), 'w')):
            with contextlib.redirect_stderr(open(os.path.join(self.mp_dir, '.err'), 'w')):
                eigval, eigvec, covmat = self.mp_get_covmat(self.mp['cosmo'], self.mp['data'], self.mp['command_line'])
        return covmat

    def get_covmat(self, config_covmat):
        self.mp['command_line'].cov = config_covmat
        with contextlib.redirect_stdout(open(os.path.join(self.mp_dir, '.out'), 'w')):
            with contextlib.redirect_stderr(open(os.path.join(self.mp_dir, '.err'), 'w')):
                eigval, eigvec, covmat = self.mp_get_covmat(self.mp['cosmo'], self.mp['data'], self.mp['command_line'])
        return covmat
    
    @property
    def varying_param_dict(self):
        return {param_name: self.mp['data'].mcmc_parameters[param_name] for param_name in self.mp['data'].get_mcmc_parameters(['varying'])}
