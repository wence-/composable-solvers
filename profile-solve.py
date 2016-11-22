from argparse import ArgumentParser
import importlib
import os
import sys
from collections import defaultdict
from functools import partial
import h5py

from firedrake import assemble, COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI

PETSc.Log.begin()

parser = ArgumentParser(description="""Profile matvecs""", add_help=False)

parser.add_argument("--problem", choices=["poisson",
                                          "elasticity",
                                          "navier_stokes",
                                          "rayleigh_benard"],
                    help="Which problem to profile")

parser.add_argument("--autorefine", action="store_true", default=False,
                    help="Refine meshes to give approximately fixed number of dofs?")

parser.add_argument("--results-directory",
                    help="Where to put the results")

parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing output?  Default is to append.")

parser.add_argument("--help", action="store_true",
                    help="Show help")

parser.add_argument("--num-matvecs", action="store", default=40,
                    type=int, help="Number of MatVecs to perform")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)


if args.problem is None:
    PETSc.Sys.Print("Must provide problem type\n")
    sys.exit(1)


if args.results_directory is None:
    PETSc.Sys.Print("Must provide results directory\n")
    sys.exit(1)


module = importlib.import_module("problem.%s" % args.problem)
problem = module.Problem()

problem.autorefine = args.autorefine


results = os.path.join(os.path.abspath(args.results_directory),
                       "Solve-timings_%s.h5" % problem.name)

solver = problem.solver()

solver.solve()

newton_its = solver.snes.getIterationNumber()
ksp_its = solver.snes.getLinearSolveIterations()

# Axes:
# - h
# - p
# - solver type

# Measured information
# - nonlinear iterations
# - linear iterations
# - time to solution
# - flops?
