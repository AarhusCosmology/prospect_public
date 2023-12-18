![prospect logo](/doc/logo.png "")

## PROSPECT: A profile likelihood code for frequentist cosmological inference
| | |
| ----- | ----- |
| Author: | Emil Brinch Holm (ebholm@phys.au.dk) |
| Documentation: | [Documentation at GitHub pages](https://aarhuscosmology.github.io/prospect/index.html) |
| Installation: | `pip install prospect-public` |
| Release paper: | [arXiv:2312.02972](https://arxiv.org/abs/2312.02972) |

## How to use

Install PROSPECT by running `pip install prospect-public` in a terminal. *Note:* PROSPECT *requires Python version 3.10 or later*. 

PROSPECT runs either fron an input yaml file or from a folder made by a previous run of PROSPECT (it will detect by itself what type of input is given). To run it from the command line, do 
```
prospect my_input_file.yaml
``` 
or 
```
prospect my_prospect_output_folder
```
In particular, you can resume PROSPECT runs that were cancelled using the latter command. 

**Run modes**: PROSPECT supports running serially, threaded or in parallel using MPI. You can specify the run mode in the input file as demonstrated in file `input/explanatory.yaml`. If you are running using MPI, remember to call PROSPECT using MPI, for example as `mpirun -n N_PROCS prospect input/example_montepython/example_montepython.yaml`. 

#### Example files

To check that PROSPECT is installed correctly, you can try running some of the example files provided in the `input/` directory.

* `example_toy/example_toy.yaml`: An analytical 30-dimensional Gaussian likelihood. Since this evaluates quickly, it is a good first test of using PROSPECT.

* `example_montepython/example_montepython.yaml`: A simple 2-parameter example with a Gaussian likelihood on the Hubble constant, showing how to interface PROSPECT with [MontePython](https://github.com/brinckmann/montepython_public). Before using this, you must set the correct `path` to your `montepython_public/montepython` directory in the yaml file `example_montepython.yaml` and specify the correct paths to your CLASS installation in `example_montepython/example.conf`. 

* `example_montepython/base2018TTTEEE.yaml`: A profile of the Hubble constant using the example file `base2018TTTEEE.param` of MontePython, which employs Planck high-ell TTTEEE and low-ell EE and TT data. Remember to also set the correct `path` in `base2018TTTEEE.yaml` as well as the correct paths to CLASS and clik in `example.conf`, as above. In this example, we have set the optimiser settings such that the profiles should converge somewhat quickly (i.e. each optimisation should take on the order of 2 hours on 8 CPU cores) such that you can use them as a starting point for your own optimisation.

* `example_cobaya/example_cobaya.yaml`: A simple example showing how to interface PROSPECT with [cobaya](https://github.com/CobayaSampler/cobaya). 

To learn how to create your own PROSPECT input files, consult `input/explanatory.yaml` which presents the possible options for input arguments. *Tip: The example files have hardcoded relative paths, so make sure to run them outside of the`input/example_*` subdirectory.*

#### Loading a profile in Python

From a PROSPECT run that has completed an analysis task, the profile likelihood data is stored in the `my_output/profile/` subdirectory of the output folder. The exact contents depend on what you choose in your input file. *Tip: You can always analyse an ongoing PROSPECT run by calling `prospect-analyse my_prospect_run` from the terminal.*

PROSPECT provides a quick tool to load the profile likelihood in Python as follows:
```
from prospect.profile import load_profile
profile_dict = load_profile('path_to_my_run')
```
whence `profile_dict` is a dictionary with the best-fitting parameter values at each point in the profile, along with the associated likelihood values. 

#### Reoptimising

Not satisfied with your profile? You can always queue new `OptimiseTask`s with the reoptimising feature. Add extra tasks from the terminal by running
```
prospect-reoptimise -y my_reoptimise_settings.yaml -o folder_to_reoptimise (--override)
```
Here, `my_reoptimise_settings.yaml` should be a yaml file similar in structure to PROSPECT input files, but only containing the "profile" substructure. For example:
```
profile:
    temperature_schedule: 'exponential'
    temperature_range: [0.2, 0.01]

    step_size_schedule: 'adaptive'
    step_size_adaptive_interval: [0.1999, 0.2001]
    step_size_adaptive_multiplier: 0.4
    step_size_adaptive_initial: 0.25

    steps_per_iteration: 1000
    max_iterations: 15
    repetitions: 3
```
If you add the `--override` flag to the command, all currently queued `OptimiseTask`s will be deleted and replaced by new ones according to the schedule chosen in `my_reoptimise_settings.yaml`. This is often recommended, since otherwise the newly added tasks may never get run. Note that you can also use this feature to change the `values` setting, allowing you to add new points to sample the profile likelihood at.

**NOTE**: PROSPECT does not have a working convergence criterion, so you must check the profile yourself to determine whether it is converged. This can be assessed in the `my_prospect_output/profile/my_prospect_output_schedule.pdf` figure. If the log-likelihoods have not changed much in the last iterations, and the acceptance rate is non-zero, either the profile is converged or needs reoptimising from a larger temperature. If you are still having issues converging your profiles, you should change some of the optimiser settings. The best settings are very problem-dependent. We suggest trying a mixture of the following:

* Increasing the `steps_per_iteration`, `max_iterations` and `repetitions`. This makes the simulated annealing MCMC more exploratory.

* Changing `temperature_range`: If your input MCMC is not well-converged, starting at larger temperatures can be an advantage. Otherwise, be careful that you have not picked a too small temperature in the second entry.

If you are still having problem converging, feel free to write me at ebholm@phys.au.dk.

#### PROSPECT snapshots

PROSPECT dumps its current state to the output folder at regular intervals given by the input file. In particular, PROSPECT output folders always contain a `status.txt` file which gives an overview of all finished, failed, queued and ongoing tasks of the run. 

PROSPECT runs can be loaded and inspected interactively in Python. To do so, from Python, import the `load` function by running `from prospect.run import load`. `load` takes the path to an output folder from a previous PROSPECT run, and returns a `Scheduler` instance whose tasks can be accessed directly. For example,
```
from prospect.run import load 
my_run = load('path_to_my_run')
all_finished_tasks = my_run.tasks.done.values() # tasks.done is a dict with task id as key and task as value
```


Check out the [documentation](https://aarhuscosmology.github.io/prospect/index.html) for more detailed instructions on installation, how to define input files, as well as example workflows.

## Having issues?
If you are experiencing problems using PROSPECT, do not hesitate to write a mail at ebholm@phys.au.dk or submit an issue on the repository here.

## How to cite 
If you are using PROSPECT for a publication, please cite the PROSPECT release paper: [arXiv:2312.02972](https://arxiv.org/abs/2312.02972). In addition, please cite the codes that PROSPECT uses in your work. This could be:
* An MCMC sampler, either [MontePython](https://github.com/brinckmann/montepython_public) or [cobaya](https://github.com/CobayaSampler/cobaya)
* Theory codes, such as [CLASS](https://github.com/lesgourg/class_public) or [CAMB](https://github.com/cmbant/CAMB)
* Data measurements and likelihoods that you use, for example the Planck 2018 data release [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
