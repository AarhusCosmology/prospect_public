![prospect logo](/doc/logo.png "")

## PROSPECT: A profile likelihood code for frequentist cosmological inference
| | |
| ----- | ----- |
| Author: | Emil Brinch Holm (ebholm@phys.au.dk) |
| Documentation: | [Documentation at GitHub pages](https://aarhuscosmology.github.io/prospect/index.html) |
| Installation: | `pip install prospect-public` (but see documentation) |
| Release paper: | *to appear on arXiv soon.* |

## How to use

Install PROSPECT by running `pip install prospect-public` in a terminal.

PROSPECT runs either fron an input yaml file or from a folder made by a previous run of PROSPECT (it will detect by itself what type of input is given). To run it from the command line, do `prospect my_input_file.yaml` or `prospect my_prospect_output_folder`. To check that PROSPECT is installed correctly, you can try running the `example_toy.yaml` model in the `input/example_toy` directory. Similar example input files can be found in `input/example_montepython` and `input/example_cobaya` for usage with either MontePython or cobaya.

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
