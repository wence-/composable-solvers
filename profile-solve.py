from argparse import ArgumentParser
import importlib
import os
import sys
import cPickle
import pandas
from collections import defaultdict

from firedrake import COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI

PETSc.Log.begin()

parser = ArgumentParser(description="""Profile solves""", add_help=False)

parser.add_argument("--problem", choices=["poisson",
                                          "elasticity",
                                          "navier_stokes",
                                          "rayleigh_benard"],
                    help="Which problem to profile")

parser.add_argument("--results-file", action="store", default="solve-timings.csv",
                    help="Where to put the results")

parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing output?  Default is to append.")

parser.add_argument("--help", action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)


if args.problem is None:
    PETSc.Sys.Print("Must provide problem type\n")
    sys.exit(1)


module = importlib.import_module("problem.%s" % args.problem)
problem = module.Problem()

results = os.path.abspath(args.results_file)

warm = defaultdict(bool)

def run_solve(problem, degree, size):
    problem.reinit(degree=degree, size=size)
    for name in problem.parameter_names:
        parameters = getattr(problem, name)
        solver = problem.solver(parameters=parameters)

        PETSc.Sys.Print("\nSolving with parameter set '%s'..." % name)
        if not warm[(name, degree)]:
            PETSc.Sys.Print("Warmup solve")
            problem.u.assign(0)
            with PETSc.Log.Stage("Warmup"):
                try:
                    solver.solve()
                except:
                    PETSc.Sys.Print("Unable to solve %s, %s, %s" % (name, problem.N, problem.degree))
            warm[(name, degree)] = True

        problem.u.assign(0)

        PETSc.Sys.Print("Timed solve")
        with PETSc.Log.Stage("P(%d, %d) Warm solve %s" % (degree, size, name)):
            try:
                solver.solve()
                snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
                ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
                pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
                pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
                jac = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
                residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
                snes_time = problem.comm.allreduce(snes["time"], op=MPI.SUM) / problem.comm.size
                jac_time = problem.comm.allreduce(jac["time"], op=MPI.SUM) / problem.comm.size
                residual_time = problem.comm.allreduce(residual["time"], op=MPI.SUM) / problem.comm.size
                ksp_time = problem.comm.allreduce(ksp["time"], op=MPI.SUM) / problem.comm.size
                pcsetup_time = problem.comm.allreduce(pcsetup["time"], op=MPI.SUM) / problem.comm.size
                pcapply_time = problem.comm.allreduce(pcapply["time"], op=MPI.SUM) / problem.comm.size

                newton_its = solver.snes.getIterationNumber()
                ksp_its = solver.snes.getLinearSolveIterations()

                if COMM_WORLD.rank == 0:
                    if not os.path.exists(os.path.dirname(results)):
                        os.makedirs(os.path.dirname(results))

                    if args.overwrite:
                        mode = "w"
                        header = True
                    else:
                        mode = "a"
                        header = not os.path.exists(results)

                    data = {"snes_its": newton_its,
                            "ksp_its": ksp_its,
                            "SNESSolve": snes_time,
                            "KSPSolve": ksp_time,
                            "PCSetUp": pcsetup_time,
                            "PCApply": pcapply_time,
                            "JacobianEval": jac_time,
                            "FunctionEval": residual_time,
                            "num_processes": problem.comm.size,
                            "mesh_size": problem.N,
                            "dimension": problem.dimension,
                            "degree": problem.degree,
                            "solver_parameters": cPickle.dumps(solver.parameters),
                            "parameter_name": name,
                            "dofs": problem.u.dof_dset.layout_vec.getSize(),
                            "name": problem.name}

                    df = pandas.DataFrame(data, index=[0])

                    df.to_csv(results, index=False, mode=mode, header=header)
            except:
                PETSc.Sys.Print("Unable to solve %s, %s, %s" % (name, problem.N, problem.degree))
        PETSc.Sys.Print("Solving with parameter set '%s'...done" % name)


if problem.dimension == 2:
    sizes = [16, 32, 64, 128, 256, 512]
    degrees = range(1, 7)
elif problem.dimension == 3:
    sizes = [4, 8, 16, 32, 64]
    degrees = range(1, 5)
else:
    raise ValueError("Unhandled dimension %d", problem.dimension)

for size in sizes:
    for degree in degrees:
        run_solve(problem, degree, size)
            
