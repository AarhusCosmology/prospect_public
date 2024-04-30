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
from prospect.tasks.base_analyse_task import BaseAnalyseTask
from prospect.tasks.optimise_task import OptimiseTask

# Switch matplotlib backend to one that doesn't pop up figures
matplotlib.use('Agg')

class AnalyseGlobalOptimisationTask(BaseAnalyseTask):
    def run(self, tasks: list[OptimiseTask]): 
        if not tasks:
            print(f"No tasks to analyse. Ending AnalyseProfileTask.")
            return
        print("Running global optimisation analysis script...")
        kernel = initialise_kernel(self.config.kernel, self.config.io.dir, self.id)
        optimise_tasks, initialise_tasks = [], []
        for task in tasks:
            if task.type == 'OptimiseTask':
                optimise_tasks.append(task)
            elif task.type == 'InitialiseOptimiserTask':
                initialise_tasks.append(task)
        bestfits = self.sort_tasks(optimise_tasks, initialise_tasks)
        self.write_results(bestfits, kernel)
        if self.config.profile.plot_schedule:
            tic = time.perf_counter()
            self.plot_schedule(optimise_tasks)
            print(f"Plotted schedule in {time.perf_counter() - tic:.5} s")

    def sort_tasks(self, optimise_tasks, initialise_tasks):
        bestfits = {}
        for idx_rep in range(self.config.profile.repetitions):
            current_bestfit = {'loglkl': np.inf}
            for task in optimise_tasks:
                if task.optimise_settings['repetition_number'] == idx_rep:
                    if task.optimiser.bestfit['loglkl'] < current_bestfit['loglkl']:
                        current_bestfit = task.optimiser.bestfit 
            bestfits[idx_rep] = current_bestfit # store the best in each rep
            best_rep = list(bestfits.keys())[np.argmin([rep['loglkl'] for rep in bestfits.values()])]
            bestfits[idx_rep]['best_rep'] = best_rep

            for initialise_task in initialise_tasks:
                # All reps are all initialised to the same point
                bestfits[idx_rep]['initial'] = initialise_task.initial_bestfit
                bestfits[idx_rep]['initial_loglkl'] = initialise_task.initial_loglkl
                bestfits[idx_rep]['covmat'] = initialise_task.initial_covmat
                break
        return bestfits

    def write_results(self, bestfits, kernel) -> None:
        results_file = os.path.join(self.dir, f'{self.config.profile.parameter}.txt')
        param_names = [param_name for param_name in kernel.param['varying'] if param_name != self.config.profile.parameter]

        print(f"Writing current optimisation results to {results_file}")
        with open(results_file,"w") as file:
            # Write header
            file.write(f"Repetition number \t -loglkl \t initial_loglkl \t ")
            file.write(f" \t ".join([str(name) for name in param_names])+"\n")
            # Write results
            for idx_rep in range(self.config.profile.repetitions):
                if bestfits[idx_rep]['loglkl'] == np.inf:
                    file.write(f"{idx_rep} \t --- no optimisations finished ---\n")
                else:
                    file.write(f"{idx_rep} \t {bestfits[idx_rep]['loglkl']:.10e} \t ")
                    file.write(f"{bestfits[idx_rep]['initial_loglkl']:.10e} \t ")
                    file.write(" \t ".join([str(np.round(bestfits[idx_rep]['position'][name][0], 10)) for name in param_names])+"\n")

    def plot_schedule(self, optimise_tasks):
        # Collect nested dict data structure
        iter_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # Sort optimisetasks by their repetition and iteration numbers
        for task in optimise_tasks:
            iter_data[task.optimise_settings['repetition_number']][task.optimise_settings['iteration_number']] = {
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

        for idx_rep in range(self.config.profile.repetitions):
            if idx_rep in iter_data:
                iterlist = list(iter_data[idx_rep].keys())
                order = np.argsort(iterlist)
                iterlist = np.array(iterlist)[order]
                loglkls = np.array([it['loglkl'] for it in iter_data[idx_rep].values()])[order]
                acceptance_rates = np.array([it['acceptance_rate'] for it in iter_data[idx_rep].values()])[order]
                step_sizes = np.array([it['step_size'] for it in iter_data[idx_rep].values()])[order]
                ax[0].plot(iterlist, loglkls, '.-', alpha=0.8, ms=self.fig['rep_ms'][idx_rep])
                ax[1].plot(iterlist, acceptance_rates, '.-', alpha=0.8, ms=self.fig['rep_ms'][idx_rep])
                ax[2].plot(iterlist, step_sizes, '.-')    
                if plot_adaptive == 'adaptive':
                    multipliers = np.array([it['step_size_change']['current_multiplier'] for it in iter_data[idx_rep].values()])[order]
                    ax[3].plot(iterlist, multipliers, '.-', alpha=0.8, ms=self.fig['rep_ms'][idx_rep])
                if len(iterlist) > current_longest['len']:
                    current_longest = {
                        'len': len(iterlist),
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
            iterlist = list(iter_data[current_longest['idx_rep']].keys())
            order = np.argsort(iterlist)
            iterlist = np.array(iterlist)[order]
            temperatures = np.array([it['temperature'] for it in iter_data[current_longest['idx_rep']].values()])[order]
            ax[2].plot(iterlist, temperatures, 'r.-', label='T')    
            handles, labels = ax[0].get_legend_handles_labels()
            if handles and labels:
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                ax[0].legend(handles, labels, fontsize=0.5*self.fig['fontsize'])
            ax[0].legend(handles, labels, fontsize=0.5*self.fig['fontsize'])
            ax[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, f'optimisation_schedule.pdf'))
        plt.close()