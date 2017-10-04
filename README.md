#### Experimentation framework for the manuscript:

> Robert C. Kirby and Lawrence Mitchell. "Solver composition across
> the PDE/linear algebra barrier".

#### Usage

To run the experiments, you will need a version
of [Firedrake](http://www.firedrakeproject.org/).  The paper
referenced above contains documentation of the versions used to
generate the results in the repository.

There are three python scripts that run the code and collect data:

- poisson-weak-scale.py
- rb-weak-scale.py
- profile-matvec.py

The usage as required to generate the data in the paper is in the
matching PBS job submission scripts.

The `results` directory contains data for the results in the paper.
The python scripts in that directory can be used to recreate the
plots, and the tables from the paper.
In addition to the processed data, this directory also contains the
output and log files (from PETSc's -log_view) for each of the runs.
Note that the FLOP-counting routine used in PyOP2 for some of the
values was buggy (overcounting by a factor of 3, and therefore some
FLOP rates in the log files are bogus).  This is corrected in the
profile-matvec.py script when logging the data used to generate
roofline plots.
