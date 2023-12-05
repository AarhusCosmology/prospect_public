![prospect logo](/doc/logo.png "")

## PROSPECT: A profile likelihood code for frequentist cosmological inference
| | |
| ----- | ----- |
| Author: | Emil Brinch Holm (ebholm@phys.au.dk) |
| Documentation: | [Documentation at GitHub pages](https://aarhuscosmology.github.io/prospect/index.html) |
| Installation: | `pip install prospect-public` (but see documentation) |
| Release paper: | *to appear on arXiv soon.* |

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

**Run modes**: PROSPECT supports running serially, threaded or in parallel using MPI. You can specify the run mode in the input file as demonstrated in file `input/explanatory.yaml`.

#### Example files

To check that PROSPECT is installed correctly, you can try running some of the example files provided in the `input/` directory.

* `example_toy`: An analytical 30-dimensional Gaussian likelihood. Since this evaluates quickly, it is a good first test of using PROSPECT.

* `example_montepython`: A simple example showing how to interface PROSPECT with [MontePython](https://github.com/brinckmann/montepython_public). 

* `example_cobaya`: A simple example showing how to interface PROSPECT with [cobaya](https://github.com/CobayaSampler/cobaya). 

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
If you add the `--override` flag to the command, all currently queued `OptimiseTask`s will be deleted and replaced by new ones according to the schedule chosen in `my_reoptimise_settings.yaml`. Note that you can also use this feature to change the `values` setting, allowing you to add new points to sample the profile likelihood at.

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
If you are using PROSPECT for a publication, please cite the PROSPECT release paper: *to appear on arXiv soon.* In addition, please cite the codes that PROSPECT uses in your work. This could be:
* An MCMC sampler, either [MontePython](https://github.com/brinckmann/montepython_public) or [cobaya](https://github.com/CobayaSampler/cobaya)
* Theory codes, such as [CLASS](https://github.com/lesgourg/class_public) or [CAMB](https://github.com/cmbant/CAMB)
* Data measurements and likelihoods that you use, for example the Planck 2018 data release [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
