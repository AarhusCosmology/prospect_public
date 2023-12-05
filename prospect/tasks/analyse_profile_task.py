from collections import defaultdict
from copy import deepcopy
import os 
import time
from typing import Type
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.interpolate import interp1d
from scipy import stats
from prospect.input import Configuration
from prospect.kernels.initialisation import initialise_kernel
from prospect.tasks.base_task import BaseTask
from prospect.tasks.optimise_task import OptimiseTask

# Switch matplotlib backend to one that doesn't pop up figures
matplotlib.use('Agg')

def argmax(iterable):
    """From https://stackoverflow.com/questions/16945518/finding-the-index-of-the-value-which-is-the-min-or-max-in-python"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]

class AnalyseProfileTask(BaseTask):
    priority = 75.0

    # Figure settings
    cm = 1/2.54  # centimeters in inches
    fig = {
        'width': 8.6*cm, # PRL figure width
        'height': 2,
        'fontsize': 11/1.1,
        'rep_colors': ['b', 'r', 'g', 'c', 'y', 'm']*50,
        'rep_ms': [4.5, 4.0, 3.5, 3, 2.5, 2., 1.5, 1., 0.5]*50,
        'interval_styles': ['--', ':']*5
    }

    def __init__(self, config: Configuration, required_task_ids: list[int]):
        super().__init__(config, required_task_ids)
        self.dir = os.path.join(config.io.dir, config.run.jobtype)

    def run(self, tasks: list[OptimiseTask]): 
        if not tasks:
            print(f"No tasks to analyse. Ending AnalyseProfileTask.")
            return
        print("Running profile analysis script...")
        kernel = initialise_kernel(self.config.kernel, self.config.io.dir, self.id)
        optimise_tasks, initialise_tasks = [], []
        param_vals = []
        for task in tasks:
            if task.type == 'OptimiseTask':
                if task.optimise_settings['fixed_param_val'] not in param_vals:
                    param_vals.append(task.optimise_settings['fixed_param_val'])
                optimise_tasks.append(task)
            elif task.type == 'InitialiseOptimiserTask':
                initialise_tasks.append(task)

        param_vals = np.sort(param_vals)
        old_param_vals = deepcopy(self.config.profile.values)
        self.config.profile.values = param_vals

        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)
        
        self.bestfits = self.sort_tasks(optimise_tasks, initialise_tasks)
        profile = self.compute_profile(self.bestfits)
        self.write_results(self.bestfits, kernel)
        self.write_profile_status(optimise_tasks, initialise_tasks)

        intervals = self.compute_intervals_neyman(profile)
        self.write_intervals(profile, intervals, kernel)

        if self.config.profile.plot_profile:
            tic = time.perf_counter()
            self.plot_profile(profile, intervals, self.bestfits, kernel)
            print(f"Plotted profile in {time.perf_counter() - tic:.5} s")
        if self.config.profile.plot_schedule:
            tic = time.perf_counter()
            self.plot_schedule(optimise_tasks)
            print(f"Plotted schedule in {time.perf_counter() - tic:.5} s")
        self.config.profile.values = old_param_vals
    
    def write_profile_status(self, optimise_tasks, initialise_tasks):
        status_file = os.path.join(self.dir, f'{self.config.profile.parameter}_status.txt')

        print(f"Writing current profile status to {status_file}")
        with open(status_file,"w") as file:
            # Write header
            file.write(f"{self.config.profile.parameter:14.10} ")
            file.write(f"\t\t | rep. no.  current iter \t current loglkl \t best iter \t best loglkl \t converged? ")
            
            for fixed_val in self.config.profile.values:
                file.write(f"\n{fixed_val:14.10} ")
                for idx_rep in range(self.config.profile.repetitions):
                    rep_tasks = [task for task in optimise_tasks if task.optimise_settings['fixed_param_val'] == fixed_val and task.optimise_settings['repetition_number'] == idx_rep]
                    if rep_tasks:
                        current_task = rep_tasks[argmax([task.optimise_settings['iteration_number'] for task in rep_tasks])]
                        current_iter = current_task.optimiser.bestfit['iteration_number']
                        current_loglkl = current_task.optimiser.bestfit['loglkl']

                        best_task = rep_tasks[argmax([-task.optimiser.bestfit['loglkl'] for task in rep_tasks])]
                        best_iter = best_task.optimiser.bestfit['iteration_number']
                        best_loglkl = best_task.optimiser.bestfit['loglkl']
                        pad_current = 13
                        pad_best = 9
                        write_string = f"| rep {idx_rep+1} \t {current_iter:<{pad_current}} \t {current_loglkl:.10e} \t {best_iter:<{pad_best}} \t {best_loglkl:.10e} \t {current_task.converged} "
                        if idx_rep == 0:
                            file.write(f"\t {write_string}")
                        else:
                            file.write(f"\n\t\t\t\t {write_string}")

    def emit_tasks(self) -> list[Type[BaseTask]]:
        # Never emit anything; the OptimiseTasks manage convergence themselves.
        return []
    
    def sort_tasks(self, optimise_tasks, initialise_tasks):
        # Order the tasks
        bestfits = {}
        for fixed_value in self.config.profile.values:
            bestfits[fixed_value] = {}
            for idx_rep in range(self.config.profile.repetitions):
                current_bestfit = {'loglkl': np.inf}
                for task in optimise_tasks:
                    if task.optimise_settings['fixed_param_val'] == fixed_value:
                        if task.optimise_settings['repetition_number'] == idx_rep:
                            if task.optimiser.bestfit['loglkl'] < current_bestfit['loglkl']:
                                current_bestfit = task.optimiser.bestfit 
                bestfits[fixed_value][idx_rep] = current_bestfit # store the best in each rep
            
            # Find the best rep for this fixed value
            best_rep = list(bestfits[fixed_value].keys())[np.argmin([rep['loglkl'] for rep in bestfits[fixed_value].values()])]
            if np.any([rep['loglkl'] < np.inf for rep in bestfits[fixed_value].values()]):
                avg_loglkl = np.mean([rep['loglkl'] for rep in bestfits[fixed_value].values() if rep['loglkl'] < np.inf])
            else:
                avg_loglkl = np.inf
            bestfits[fixed_value]['best_rep'] = best_rep
            bestfits[fixed_value]['avg_loglkl'] = avg_loglkl

            for initialise_task in initialise_tasks:
                if initialise_task.fixed_param_val == fixed_value:
                    # All reps are all initialised to the same point
                    bestfits[fixed_value]['initial'] = initialise_task.initial_bestfit
                    bestfits[fixed_value]['initial_loglkl'] = initialise_task.initial_loglkl
                    bestfits[fixed_value]['covmat'] = initialise_task.initial_covmat
                    break
        return bestfits
    
    def compute_profile(self, bestfit_dict):
        """
            Default analysis method is to fit the best point of the profile and its two 
            neighbouring points to a parabola and take the top point of the parabola 
            as the bestfit chi2
        
        """

        # order points in terms of profile parameter
        param_vals, loglkls = [], []
        for param_val, bestfit in bestfit_dict.items():
            param_vals.append(param_val)
            loglkl = bestfit[bestfit['best_rep']]['loglkl']
            if loglkl < np.inf:
                loglkls.append(loglkl)
            else:
                # No optimisations finished, take the initial values
                loglkls.append(bestfit['initial_loglkl'])
        ordering = np.argsort(param_vals)
        param_vals = np.array(param_vals)[ordering]
        loglkls = np.array(loglkls)[ordering]

        best_index = np.argmin(loglkls)
        if best_index == 0 or best_index == len(loglkls) - 1:
            # best point is on one of the edges of the profile
            global_best_param_val = param_vals[best_index]
            global_best_loglkl = loglkls[best_index]
        else:
            # find the three parabola points to get Delta chi2
            x = [param_vals[idx] for idx in [best_index-1, best_index, best_index+1]]
            f = [loglkls[idx] for idx in [best_index-1, best_index, best_index+1]]
            global_best_param_val, global_best_loglkl = self.get_bestfit_parabola(x, f)
            
        for param_val, bestfit in bestfit_dict.items():
            for idx_rep in range(self.config.profile.repetitions):
                if idx_rep in bestfit:
                    bestfit[idx_rep]['Delta_chi2'] = 2*(bestfit[idx_rep]['loglkl'] - global_best_loglkl)
        
        Delta_chi2 = []
        for param_val, bestfit in bestfit_dict.items():
            loglkl = bestfit[bestfit['best_rep']]['loglkl']
            if loglkl < np.inf:
                Delta_chi2.append(bestfit[bestfit['best_rep']]['Delta_chi2'])
            else:
                Delta_chi2.append(2*(bestfit['initial_loglkl'] - global_best_loglkl))
        Delta_chi2 = np.array(Delta_chi2)[ordering]

        profile = {
            'global_best_loglkl': global_best_loglkl,
            'global_best_param_val': global_best_param_val,
            'param_vals': param_vals,
            'loglkls': loglkls,
            'Delta_chi2': Delta_chi2
        }
        return profile

    def write_results(self, bestfit_dict, kernel) -> None:
        results_file = os.path.join(self.dir, f'{self.config.profile.parameter}.txt')
        param_names = [param_name for param_name in kernel.param['varying'] if param_name != self.config.profile.parameter]

        print(f"Writing current profile results to {results_file}")
        with open(results_file,"w") as file:
            # Write header
            file.write(f"{self.config.profile.parameter} \t Delta_chi2 \t -loglkl \t avg_best_loglkl \t initial_loglkl \t ")
            file.write(f" \t ".join([str(name) for name in param_names])+"\n")
            # Write results
            for fixed_val, bestfits_fixed_val in bestfit_dict.items():
                bestfit = bestfits_fixed_val[bestfits_fixed_val['best_rep']]
                if bestfit['loglkl'] == np.inf:
                    file.write(f"{fixed_val:.10f} \t --- no optimisations finished ---\n")
                else:
                    file.write(f"{fixed_val:.10f} \t {bestfit['Delta_chi2']:.10e} \t {bestfit['loglkl']:.10e} \t ")
                    file.write(f"{bestfits_fixed_val['avg_loglkl']:.10e} \t {bestfits_fixed_val['initial_loglkl']:.10e} \t ")
                    file.write(" \t ".join([str(np.round(bestfit['position'][name][0], 10)) for name in param_names])+"\n")

    def compute_intervals_neyman(self, profile):
        """
            Computes confidence intervals using the Neyman construction

        """
        intervals = {}

        def interp_onto(xdata, ydata, new_xdata, kind='linear'):
            return interp1d(xdata, ydata, fill_value="extrapolate", kind=kind)(new_xdata)
        param_cont = np.linspace(np.min(profile['param_vals']), np.max(profile['param_vals']), 10000)
        chi2_interped = interp_onto(profile['param_vals'], profile['Delta_chi2'], param_cont, kind='linear')
        idx_min = np.argmin(np.abs(param_cont - profile['global_best_param_val']))

        for cl in self.config.profile.confidence_levels:
            chi2 = stats.chi2.ppf(cl, 1) 
            intervals[cl] = [None, None]
            if idx_min > 0:
                lower_idx = np.argmin(np.abs(chi2_interped[0:idx_min] - chi2))
                intervals[cl][0] = param_cont[lower_idx]
            if idx_min < len(param_cont) - 1:
                upper_idx = idx_min + np.argmin(np.abs(chi2_interped[idx_min:-1] - chi2))
                intervals[cl][1] = param_cont[upper_idx]
        return intervals
    
    def write_intervals(self, profile, intervals, kernel) -> None:
        interval_file = os.path.join(self.dir, f'{self.config.profile.parameter}_intervals.txt')
        print(f"Writing current profile intervals to {interval_file}...")
        with open(interval_file, "w") as file:
            file.write('C.L. \t C.I. \t bestfit (+dist. to upper bound)(-dist. to lower bound)\n')
            for cl, interval in intervals.items():
                file.write(f"{cl*100} %: \t {interval} \t {profile['global_best_param_val']}")
                upper_dist, lower_dist = None, None
                if interval[1] is not None:
                    upper_dist = interval[1] - profile['global_best_param_val']
                if interval[0] is not None:
                    lower_dist = profile['global_best_param_val'] - interval[0]
                file.write(f"(+{upper_dist})(-{lower_dist}) \n")
                

    def plot_profile(self, profile, intervals, bestfit_dict, kernel) -> None:
        param_vals = profile['param_vals']
        loglkls    = profile['loglkls']
        Delta_chi2 = profile['Delta_chi2']

        fig, ax = plt.subplots(1, 1, figsize=(self.fig['width'], self.fig['height']))
        ax.set_xlabel(self.config.profile.parameter, fontsize=self.fig['fontsize'])
        if self.config.profile.plot_Delta_chi2:
            ax.set_ylabel('Delta chi2', fontsize=self.fig['fontsize'])
            for idx, cl in enumerate(self.config.profile.confidence_levels):
                chi2 = stats.chi2.ppf(cl, 1)
                ax.plot(param_vals, chi2*np.ones(len(param_vals)), 'k', linestyle=self.fig['interval_styles'][idx], lw=0.5, label=f'{cl} C.L.')
                for bound in intervals[cl]:
                    if bound is not None:
                        ax.plot([bound, bound], [ax.get_ylim()[0], chi2], 'k', linestyle=self.fig['interval_styles'][idx], lw=0.7)
            ax.plot(param_vals, Delta_chi2, 'k.-', lw=0.9, label='current', zorder=2)
        else:
            ax.set_ylabel('-loglkl', fontsize=self.fig['fontsize'])
            ax.plot(param_vals, loglkls, 'k.-', lw=0.9, label='current', zorder=2)

        if self.config.profile.detailed_plot:
            # Plot initial profile
            param_vals, loglkls_initial = [], []
            for param_val, bestfit in bestfit_dict.items():
                param_vals.append(param_val)
                loglkls_initial.append(bestfit['initial_loglkl'])
            ordering = np.argsort(param_vals)
            param_vals = np.array(param_vals)[ordering]
            loglkls_initial = np.array(loglkls_initial)[ordering]
            if self.config.profile.plot_Delta_chi2:
                ax.plot(param_vals, 2*(loglkls_initial - profile['global_best_loglkl']), 'y.--', ms=5, lw=0.7, alpha=0.8, label='initial')
            else:
                ax.plot(param_vals, loglkls_initial, 'y.--', ms=5, lw=0.7, alpha=0.8, label='initial', zorder=3)
            
            # Plot points from each rep
            for param_val, bestfit in bestfit_dict.items():
                for idx_rep in range(self.config.profile.repetitions):
                    if idx_rep in bestfit: # and idx_rep != bestfit['best_rep']:
                        if self.config.profile.plot_Delta_chi2:
                            ax.plot(param_val, 2*(bestfit[idx_rep]['loglkl'] - profile['global_best_loglkl']), '.', alpha=0.5, ms=self.fig['rep_ms'][idx_rep], color=self.fig['rep_colors'][idx_rep])
                        else:
                            ax.plot(param_val, bestfit[idx_rep]['loglkl'], '.', alpha=0.5, ms=self.fig['rep_ms'][idx_rep], color=self.fig['rep_colors'][idx_rep])
            for idx_rep in range(self.config.profile.repetitions):
                ax.plot([], [], '.', alpha=0.5, ms=self.fig['rep_ms'][idx_rep], color=self.fig['rep_colors'][idx_rep], label=f"rep {idx_rep}")
            
            if self.config.kernel.type == 'analytical':
                # Compute profile with scipy for comparison
                fixed_param_range = [min(self.config.profile.values), max(self.config.profile.values)]
                fixed_param_cont  = np.linspace(*fixed_param_range, 20)
                kernel.set_fixed_parameters({self.config.profile.parameter: 0.0})
                scipy_profile = []
                for fixed_val in fixed_param_cont:
                    scipy_profile.append(kernel.get_scipy_profile(self.config.profile.parameter, fixed_val))
                if self.config.profile.plot_Delta_chi2:
                    ax.plot(fixed_param_cont, 2*(scipy_profile - profile['global_best_loglkl']), 'g--', lw=1.5, label='scipy')
                else:
                    ax.plot(fixed_param_cont, scipy_profile, 'g--', lw=1.5, label='scipy')
        
        #ax.legend(fontsize=self.fig['fontsize']*0.8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, f'{self.config.profile.parameter}.pdf'))
        plt.close()

    def get_bestfit_parabola(self, x_triple, f_triple):
        # pm, pj and pp are the points (ordered according to their first coordinates) within which the bestfit lies
        # interpolating polynomial taken from the implementation of https://github.com/mwt5345/class_ede/issues/1
        xm, xj, xp = x_triple[0], x_triple[1], x_triple[2]
        fm, fj, fp = f_triple[0], f_triple[1], f_triple[2]
        x_best = (fp*(xj - xm)*(xj + xm) + fj*(xm - xp)*(xm + xp) + fm*(-pow(xj,2) + pow(xp,2)))/(2.*(fp*(xj - xm) + fj*(xm - xp) + fm*(-xj + xp)))
        f_best = fm + pow(-(fp*pow(xj - xm,2)) + fj*pow(xm - xp,2) + fm*(xj - xp)*(xj - 2*xm + xp),2)/(4.*(xj - xm)*(xj - xp)*(xm - xp)*(fp*(-xj + xm) + fm*(xj - xp) + fj*(-xm + xp)))
        return [x_best, f_best]

    def plot_schedule(self, optimise_tasks):
        # Collect nested dict data structure
        iter_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
        for task in optimise_tasks:
            iter_data[task.optimise_settings['fixed_param_val']][task.optimise_settings['repetition_number']][task.optimise_settings['iteration_number']] = {
                'acceptance_rate': task.optimiser.bestfit['acceptance_rate'],
                'temperature': task.optimiser.settings['temperature'],
                'step_size': task.optimiser.settings['step_size'],
                'step_size_change': task.optimiser.settings['step_size_change'],
                'loglkl': task.optimiser.bestfit['loglkl']
            }
        plot_adaptive = self.config.profile.step_size_schedule == 'adaptive' and self.config.profile.step_size_adaptive_multiplier == 'adaptive'

        if plot_adaptive == 'adaptive':
            fig, ax = plt.subplots(4, 1, figsize=(self.fig['width'], 3*self.fig['height']))
        else:
            fig, ax = plt.subplots(3, 1, figsize=(self.fig['width'], 2.5*self.fig['height']))
        current_longest = {'len': 0}
        for fixed_val in self.config.profile.values:
            # Actually: Consider plotting the average over reps!
            for idx_rep in range(self.config.profile.repetitions):
                if fixed_val in iter_data:
                    if idx_rep in iter_data[fixed_val]:
                        iterlist = list(iter_data[fixed_val][idx_rep].keys())
                        order = np.argsort(iterlist)
                        iterlist = np.array(iterlist)[order]
                        loglkls = np.array([it['loglkl'] for it in iter_data[fixed_val][idx_rep].values()])[order]
                        acceptance_rates = np.array([it['acceptance_rate'] for it in iter_data[fixed_val][idx_rep].values()])[order]
                        step_sizes = np.array([it['step_size'] for it in iter_data[fixed_val][idx_rep].values()])[order]
                        ax[0].plot(iterlist, loglkls, '.-', alpha=0.8, ms=self.fig['rep_ms'][idx_rep])
                        ax[1].plot(iterlist, acceptance_rates, '.-', alpha=0.8, ms=self.fig['rep_ms'][idx_rep])
                        ax[2].plot(iterlist, step_sizes, '.-')    
                        if plot_adaptive == 'adaptive':
                            multipliers = np.array([it['step_size_change']['current_multiplier'] for it in iter_data[fixed_val][idx_rep].values()])[order]
                            ax[3].plot(iterlist, multipliers, '.-', alpha=0.8, ms=self.fig['rep_ms'][idx_rep])
                        if len(iterlist) > current_longest['len']:
                            current_longest = {
                                'len': len(iterlist),
                                'fixed_val': fixed_val,
                                'idx_rep': idx_rep
                            }
        iterlim = [-0.1, current_longest['len'] + 0.1]
        ax[0].set(ylabel='-loglkl', xlim=iterlim, xticks=[], yscale='log')
        ax[1].set(ylabel='acceptance rate', xlim=iterlim, xticks=[])
        ax[2].set(ylabel='step size & temp', xlim=iterlim, xticks=[], yscale='log')
        if plot_adaptive:
            ax[3].set(xlabel='iteration', ylabel='step multiplier', xlim=iterlim)
        if current_longest['len'] > 0:
            # Plot temperature of the run with most iterations
            iterlist = list(iter_data[current_longest['fixed_val']][current_longest['idx_rep']].keys())
            order = np.argsort(iterlist)
            iterlist = np.array(iterlist)[order]
            temperatures = np.array([it['temperature'] for it in iter_data[current_longest['fixed_val']][current_longest['idx_rep']].values()])[order]
            ax[2].plot(iterlist, temperatures, 'r.-', label='T')    
            handles, labels = ax[0].get_legend_handles_labels()
            if handles and labels:
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                ax[0].legend(handles, labels, fontsize=0.5*self.fig['fontsize'])
            ax[0].legend(handles, labels, fontsize=0.5*self.fig['fontsize'])
            ax[2].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, f'{self.config.profile.parameter}_schedule.pdf'))
        plt.close()