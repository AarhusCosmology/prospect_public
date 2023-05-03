from prospect.kernels.base_kernel import BaseKernel

class MontePythonKernel(BaseKernel):
    def initialise(self, config_kernel):
        from montepython_public.montepython.data import Data
        import montepython_public.montepython.mcmc as mcmc 
        # Create a MontePython command_line from the contents in config 

    def loglkl(self, position):
        # wrap around mcmc.chain()
        pass
