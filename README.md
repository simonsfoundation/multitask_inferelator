# Multitask Inferelator

[![Travis](https://api.travis-ci.org/simonsfoundation/inferelator_ng.svg?branch=master)](https://travis-ci.org/simonsfoundation/inferelator_ng)

[![codecov](https://codecov.io/gh/simonsfoundation/inferelator_ng/branch/master/graph/badge.svg)](https://codecov.io/gh/simonsfoundation/inferelator_ng)

To install the python packages needed for the inferelator, run `pip install -r requirements.txt`.

To install, run `python setup.py install`.

To run a workflow, run the corresponding workflow_runner script.

Note: use python 2.

----------
### Running in a serial mode:

There are two main examples for the multitask version of the inferelator (AMuSR): *B. subtilis* and yeast.

For *B. subtilis*:

`bash inferelator_runner_slurmless.sh bsubtilis_amusr_workflow_runner.py`


For yeast:

`bash inferelator_runner_slurmless.sh yeast_amusr_workflow_runner.py`

----------

### Running in parallel mode:

This is how you would run in parallel in a cluster (SLURM). Note that you can change the number of processes (as described by -n).

`python -m kvsstcp.kvsstcp --execcmd 'srun -n '${SLURM_NTASKS}' python bsubtilis_amusr_workflow_runner.py'`.


It is also possible to run in parallel locally. Below, we specify that we want to run 8 parallel processes at once (the argument after fauxSrun).

`time python -m kvsstcp.kvsstcp --execcmd 'inferelator_ng/fauxSrun 8 python bsubtilis_amusr_workflow_runner.py'`.
