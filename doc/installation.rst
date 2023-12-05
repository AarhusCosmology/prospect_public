Installation
=====================

Hi from installation.



## How to install:

` pip install . `

If developing the code, remember to install in editor mode:

` pip install -e . ` 

This will propagate your changes to the source code without need for reinstalling. 


## How to run:

PROSPECT runs either from an input yaml file or from a folder made by a previous run of PROSPECT. It will detect on its own what type of input is given.

Additionally, PROSPECT can be run either from the command line or interactively in Python.

To run from the command line:

` prospect input/debug.yaml `

To run from inside a Python script:

` from prospect.run import run; run(input) ` 

where `input` is a string pointing to either the yaml input file or an output folder of a previous PROSPECT run.
