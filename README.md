![prospect logo](/doc/logo.png "")

## PROSPECT: A profile likelihood code for frequentist cosmological inference
| | |
| ----- | ----- |
| Author: | Emil Brinch Holm (ebholm@phys.au.dk) |
| Documentation: | [Documentation at GitHub pages](https://aarhuscosmology.github.io/prospect/index.html) |
| Installation: | `pip install prospect-public` (but see documentation) |
| Release paper: | *to appear on arXiv soon.* |

## How to use

Install PROSPECT by running `pip install prospect-public` in a terminal. *Note: PROSPECT requires Python version 3.10 or later*. 

PROSPECT runs either fron an input yaml file or from a folder made by a previous run of PROSPECT (it will detect by itself what type of input is given). To run it from the command line, do 
```prospect my_input_file.yaml``` 
or 
```prospect my_prospect_output_folder```

### Example files

To check that PROSPECT is installed correctly, you can try running some of the example files provided in the `input/` directory.

* `example_toy`: An analytical 30-dimensional Gaussian likelihood. Since this evaluates quickly, it is a good first test of using PROSPECT.

* `example_montepython`: A simple example showing how to interface PROSPECT with [MontePython](https://github.com/brinckmann/montepython_public). 

* `example_cobaya`: A simple example showing how to interface PROSPECT with [cobaya](https://github.com/CobayaSampler/cobaya). 

 To learn how to create your own PROSPECT input files, consult `input/explanatory.yaml` which presents the possible options for input arguments. 

### Loading a profile in Python
From a PROSPECT run that has completed an analysis task, the profile likelihood data is stored in the `my_output/profile/` subdirectory of the output folder. PROSPECT provides a quick tool to load the profile likelihood in Python as follows:
```
from prospect.profile import load_profile
profile_dict = load_profile('path_to_my_run')
```
whence `profile_dict` is a dictionary with the best-fitting parameter values at each point in the profile, along with the associated likelihood values. 

### Loading PROSPECT snapshots
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
