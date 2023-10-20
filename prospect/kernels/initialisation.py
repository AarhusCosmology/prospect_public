def initialise_kernel(config_kernel, output_folder, task_id):
    if config_kernel.type == 'montepython':
        from prospect.kernels.montepython_kernel import MontePythonKernel
        kernel = MontePythonKernel(config_kernel, task_id, output_folder=output_folder)
    elif config_kernel.type == 'cobaya':
        from prospect.kernels.cobaya_kernel import CobayaKernel
        kernel = CobayaKernel(config_kernel, task_id, output_folder=output_folder)
    elif config_kernel.type == 'analytical':
        from prospect.kernels.analytical_kernel import AnalyticalKernel
        kernel = AnalyticalKernel(config_kernel, task_id, output_folder=output_folder)
    else:
        raise ValueError('You have specified a non-existing kernel type.')
    return kernel
