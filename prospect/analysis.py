import os
import numpy as np 
from getdist import loadMCSamples, plots
import contextlib
from prospect.mcmc import Chain

def get_gelman_rubin(chain_list: list[Chain]) -> float:
    """Returns the R-1 Gelman Rubin convergence statistic."""
    remove_burnin(chain_list)

    # Possibly split chains into subchains; although MP doesn't do this I don't think
    param_names = chain_list[0].positions.keys()
    N_chains = len(chain_list)
    Rm1 = []
    for param_name in param_names:
        total_mean = np.mean([np.mean(np.stack(chain.positions)[param_name]) for chain in chain_list])
        var_between, var_within = 0.0, 0.0
        for chain in chain_list:
            positions = np.stack(chain.positions[param_name])
            N_points = positions.shape[0]
            var_between += (np.mean(positions) - total_mean)**2/(N_chains - 1)
            var_within += N_points/(N_points - 1)/N_chains*np.mean((positions - np.mean(positions))**2)
        Rm1.append(var_between/var_within)
    # Not using square root; same convention as MontePython and CosmoMC
    return np.max(Rm1)

def getdist_gelman_rubin(chain_folder: str) -> float:
    # Suppress console output from GetDist
    with contextlib.redirect_stdout(None):
        with contextlib.redirect_stderr(None):
            samples = loadMCSamples(chain_folder)
            return samples.getGelmanRubin()

def remove_burnin(chain_list: list[Chain]) -> None:
    pass

def get_effective_sample_size():
    pass

def compute_param_covariance():
    pass

def analyse_mcmc(output_dir, jobname) -> None:
    """Makes triangle plots and intervals from MCMC chains"""
    analysis_dir = os.path.join(output_dir, 'analysis')
    if not os.path.isdir(analysis_dir):
        print(f"Directory {analysis_dir} not found, creating...")
        os.makedirs(analysis_dir)
    with contextlib.redirect_stdout(open(os.path.join(analysis_dir, "getdist.out"), 'w')):
        with contextlib.redirect_stderr(open(os.path.join(analysis_dir, "getdist.err"), 'w')):
            try:
                samples = loadMCSamples(os.path.join(output_dir, jobname))
                with open(os.path.join(analysis_dir, f"{jobname}.stats"), 'w') as f:
                    print(samples.getMargeStats(), file=f)
                    f.write(f"R-1: {getdist_gelman_rubin(os.path.join(output_dir, jobname))}")
            except Exception as e:
                raise ValueError(f'Could not analyse chain. Try increasing amount of steps or time per iteration. Original exception:\n{e}')
            g = plots.get_subplot_plotter()
            g.triangle_plot(samples, filled=True)
            g.export(os.path.join(output_dir, 'analysis', f"{jobname}_triangle.pdf"))
