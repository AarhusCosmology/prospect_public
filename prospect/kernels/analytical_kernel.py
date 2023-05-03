import numpy as np 
import os
import yaml
from prospect.kernels.base_kernel import BaseKernel

class AnalyticalKernel(BaseKernel):
    # Analytical likelihoods for development, testing and benchmarking purposes
    def initialise(self, kernel_param):
        # Instead, I should make the param-file Python-readable to easily make large dims etc.
        self.param = yaml.full_load(open(os.path.join(os.getcwd(), kernel_param), 'r'))

        if self.param['function'] == 'gaussian':
            self.loglkl = self.Gaussian
        else:
            raise ValueError('No analytical kernel with the desired function type.')
        self.dimension = self.param['dimension']
    
    def Gaussian(self, position):
        loglkl = 0.5*np.dot(position - self.param['mean'], np.matmul(np.linalg.inv(self.param['covmat']), position - self.param['mean']))
        return loglkl
    
    def loglkl(self, position):
        return self.loglkl(position)

