from prospect.tasks import OptimizeTask

def ManualSampling(config_sampling_strategy):
    samples = []
    for value in config_sampling_strategy['parameter_values']:
            for idx_walker in range(config_sampling_strategy['N_walkers']):
                samples.append(value)
    return samples

def GaussianProcess(config_sampling_strategy):
     raise NotImplementedError('')
