import os
import pickle
import yaml
import numpy as np 
import scipy as sc
from prospect.kernels.base_kernel import BaseKernel

class AnalyticalKernel(BaseKernel):
    # Analytical likelihoods for development, testing and benchmarking purposes
    def initialise(self, config_kernel, output_folder):
        # Instead, I should make the param-file Python-readable to easily make large dims etc.
        self.config = yaml.full_load(open(os.path.join(os.getcwd(), config_kernel.param), 'r'))

        if self.config['function'] == 'gaussian':
            self._loglkl = self.Gaussian
        elif self.config['function'] == 'random_gaussian':
            self._loglkl = self.Gaussian
            self.set_parameter_dict = self.set_parameter_dict_random

            settings_file = os.path.join(output_folder, f'analytical/random_gauss.pkl')
            if not os.path.isfile(settings_file):
                if 'load' in self.config:
                    # Load from user specified setting
                    print(f"Loading random Gaussian kernel from {self.config['load']}...")
                    with open(self.config['load'], 'rb') as file:
                        kernel_dict = pickle.load(file)
                else:
                    # Generate random multivariate Gaussian parameters
                    print("Generating new random Gaussian kernel...")
                    self.config['means'] = {}
                    for idx_param in range(1, self.config['dimension']+1):
                        self.config['means'][f'x{idx_param}'] = np.random.rand()
                    covmat_diag = np.diag(np.random.rand(self.config['dimension']))
                    covmat_offdiag = np.random.rand(self.config['dimension'], self.config['dimension'])
                    self.config['covmat'] = self.config['std_scale']*(covmat_diag + covmat_offdiag*self.config['off_diag_factor'])
                    kernel_dict = {
                        'means': self.config['means'],
                        'covmat': self.config['covmat'],
                        'dimension': self.config['dimension']
                    }
                # Save to disk
                with open(settings_file, 'wb') as file:
                    print(f"Saving new random Gaussian kernel to {settings_file}...")
                    pickle.dump(kernel_dict, file)
            else:
                # Load the previously generated random kernel 
                with open(settings_file, 'rb') as file:
                    kernel_dict = pickle.load(file)
            self.config['means'] = kernel_dict['means']
            self.config['covmat'] = kernel_dict['covmat']
            self.config['dimension'] = kernel_dict['dimension']
        else:
            raise ValueError('No analytical kernel with the desired function type.')
        self.dimension = self.config['dimension']
        self.scipy_profile = {}
        self.config_kernel = config_kernel
        self.output_folder = output_folder
    
    def Gaussian(self, position):
        residual = []
        for param_name, param in self.param['varying'].items():
            residual.append(position[param_name][0] - self.config['means'][param_name])
        for param_name, param in self.param['fixed'].items():
            residual.append(param['fixed_value'] - self.config['means'][param_name])
        loglkl = 0.5*np.dot(residual, np.matmul(np.linalg.inv(self.config['covmat']), residual))
        return loglkl
    
    def set_parameter_dict(self):
        for param_name, prior in self.config['param_dict'].items():
            self.param['varying'][param_name] = {'range': prior}
    
    def _loglkl(self, position):
        raise NotImplementedError("Method '_loglkl' of AnalyticalKernel must be set on initialisation!")
    
    def logprior(self, position):
        return self.log_uniform_prior(position)

    def get_initial_position(self, config_initial_position=None):
        if config_initial_position is not None:
            raise KeyError('The analytical kernel does not currently take an initial position as argument.')
        else:
            # Default
            out = {param: [0.] for param in self.param['varying'].keys()}
            for param_name, param_dict in self.param['fixed']:
                out[param_name] = [param_dict['fixed_value']]
            return out

    def get_default_initial_position(self):
        out = {param: [0.] for param in self.param['varying'].keys()}
        for param_name, param_dict in self.param['fixed']:
            out[param_name] = [param_dict['fixed_value']]
        return out

    def read_initial_position(self, config_initial_position):
        # Input should be formatted as something that can be cast to a numpy array 
        return np.array(config_initial_position)

    def get_default_covmat(self):
        # Make a diagonal covmat with 1/20 of the prior widths as stds in each parameter
        stds = []
        for param_name, param in self.param['varying'].items():
            stds.append(param['range'][1] - param['range'][0])
        return 0.005*np.diag(stds)

    def read_covmat(self, config_covmat):
        # Input should be formatted as something that can be cast to a numpy array 
        return np.array(config_covmat)

    def set_parameter_dict_random(self):
        # Overloads the set_parameter_dict from BaseKernel
        for idx_dim in range(1,self.config['dimension']+1):
            self.param['varying'][f'x{idx_dim}'] = {
                'range': [-2.5, 2.5]
            }
    
    def get_scipy_profile(self, parameter, fixed_val):
        if fixed_val not in self.scipy_profile:
            assert parameter not in self.param['varying']
            self.param['fixed'][parameter]['fixed_value'] = fixed_val
            loglkl_scipy = sc.optimize.minimize(self.wrapped_loglkl, np.zeros(len(self.param['varying']))).fun
            self.scipy_profile[fixed_val] = loglkl_scipy
        return self.scipy_profile[fixed_val]
    
    def wrapped_loglkl(self, array_of_positions: np.ndarray):
        position = {}
        for idx, param_name in enumerate(list(self.param['varying'].keys())):
            position[param_name] = [array_of_positions[idx]]
        return self.loglkl(position)