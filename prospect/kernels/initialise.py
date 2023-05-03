from prospect.kernels.analytical_kernel import AnalyticalKernel
from prospect.kernels.montepython_kernel import MontePythonKernel

def initialise_kernel(config_kernel):
    if config_kernel.type == 'montepython':
        kernel = MontePythonKernel(config_kernel.param)
    elif config_kernel.type == 'cobaya':
        raise NotImplementedError('')
    elif config_kernel.type == 'analytical':
        kernel = AnalyticalKernel(config_kernel.param)
    else:
        raise ValueError('You have specified a non-existing kernel type.')
    return kernel
