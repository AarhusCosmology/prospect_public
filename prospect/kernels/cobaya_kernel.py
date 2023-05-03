from prospect.kernels.base_kernel import BaseKernel

class CobayaKernel(BaseKernel):
    def initialise(self, config_kernel):
        raise NotImplementedError('cobaya kernel not yet implemented!')

    def loglkl(self, position):
        pass