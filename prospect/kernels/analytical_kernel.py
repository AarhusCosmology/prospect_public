import numpy as np 
import os
import yaml
from prospect.kernels.base_kernel import BaseKernel

class AnalyticalKernel(BaseKernel):
    # Analytical likelihoods for development, testing and benchmarking purposes
    def initialise(self, config_kernel, output_folder):
        # Instead, I should make the param-file Python-readable to easily make large dims etc.
        self.config = yaml.full_load(open(os.path.join(os.getcwd(), config_kernel.param), 'r'))

        if self.config['function'] == 'gaussian':
            self.loglkl = self.Gaussian
        else:
            raise ValueError('No analytical kernel with the desired function type.')
        self.dimension = self.config['dimension']
    
    def Gaussian(self, position):
        position_array = np.array(list(position.values()))
        loglkl = 0.5*np.dot(position_array - self.config['mean'], np.matmul(np.linalg.inv(self.config['covmat']), position_array - self.config['mean']))
        return loglkl
    
    def set_parameter_dict(self):
        self.param = {param_name: {'prior': interval} for param_name, interval in self.config['param_dict'].items()}
    
    def loglkl(self, position):
        raise NotImplementedError("Method 'loglkl' of AnalyticalKernel must be set on initialisation!")

    def get_initial_position(self, config_initial_position):
        raise KeyError('The analytical kernel does not currently take an initial position as argument.')

    def get_default_initial_position(self):
        return {param: 0. for param in self.param.keys()}
    
    def get_covmat(self, config_covmat):
        return config_covmat
    
    def get_default_covmat(self):
        return np.array([[0.1, 0.0], [0.0, 0.1]])