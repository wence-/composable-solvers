from argparse import ArgumentParser
import os
import cPickle
import pandas
from collections import defaultdict
import problem.poisson as module

from firedrake import COMM_WORLD, parameters
from firedrake.petsc import PETSc
from mpi4py import MPI

parameters["pyop2_options"]["lazy_evaluation"] = False

parser = ArgumentParser(description="""Profile solves""", add_help=False)

parser.add_argument("--results-file", action="store", default="solve-timings.csv",
                    help="Where to put the results")

parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing output?  Default is to append.")

parser.add_argument("--parameters", default=None, action="store",
                    help="Select specific parameter set?")

parser.add_argument("--help", action="store_true",
                    help="Show help")
parser.add_argument("--degree", action="store", default=1,
                    type=int, help="degree of problem")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)


problem_cls = module.Problem

if args.parameters is not None:
    if args.parameters not in problem_cls.parameter_names:
        raise ValueError("Unrecognised parameter '%s', not in %s", args.parameters,
                         problem_cls.parameter_names)
    parameter_names = [args.parameters]
else:
    parameter_names = problem_cls.parameter_names

results = os.path.abspath(args.results_file)

warm = defaultdict(bool)

PETSc.Log.begin()


def run_solve(problem_cls, degree, size):
    size, ref = size

    problem = problem_cls(degree=degree, N=size, refinements=ref)
    for name in parameter_names:
        parameters = getattr(problem, name)
        solver = problem.solver(parameters=parameters)
        PETSc.Sys.Print("\nSolving with parameter set %s, %s, %s..." % (name, problem.N, problem.degree))
        if not warm[(name, degree)]:
            PETSc.Sys.Print("Warmup solve")
            problem.u.assign(0)
            with PETSc.Log.Stage("Warmup"):
                solver.solve()
            warm[(name, degree)] = True

        parameters["pc_hypre_boomeramg_print_statistics"] = False
        solver = problem.solver(parameters=parameters)
        problem.u.assign(0)

        PETSc.Sys.Print("Timed solve")
        solver.snes.setConvergenceHistory()
        solver.snes.ksp.setConvergenceHistory()
        with PETSc.Log.Stage("P(%d, %d) Warm solve %s" % (degree, size, name)):
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

            num_cells = problem.comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)
            if COMM_WORLD.rank == 0:
                if not os.path.exists(os.path.dirname(results)):
                    os.makedirs(os.path.dirname(results))

                if args.overwrite:
                    mode = "w"
                    header = True
                else:
                    mode = "a"
                    header = not os.path.exists(results)
                    snes_history, linear_its = solver.snes.getConvergenceHistory()
                    ksp_history = solver.snes.ksp.getConvergenceHistory()
                data = {"snes_its": newton_its,
                        "ksp_its": ksp_its,
                        "snes_history": cPickle.dumps(snes_history),
                        "linear_its": cPickle.dumps(linear_its),
                        "ksp_history": cPickle.dumps(ksp_history),
                        "SNESSolve": snes_time,
                        "KSPSolve": ksp_time,
                        "PCSetUp": pcsetup_time,
                        "PCApply": pcapply_time,
                        "JacobianEval": jac_time,
                        "FunctionEval": residual_time,
                        "num_processes": problem.comm.size,
                        "mesh_size": problem.N * (2**ref),
                        "num_cells": num_cells,
                        "dimension": problem.dimension,
                        "degree": problem.degree,
                        "solver_parameters": cPickle.dumps(solver.parameters),
                        "parameter_name": name,
                        "dofs": problem.u.dof_dset.layout_vec.getSize(),
                        "name": problem.name}

                df = pandas.DataFrame(data, index=[0])

                df.to_csv(results, index=False, mode=mode, header=header)
        PETSc.Sys.Print("Solving with parameter set %s, %s, %s...done" % (name, problem.N, problem.degree))


degree = args.degree
if degree == 1:
    # Degree 1 mesh sizes:
    target_mesh_size = {24: (50, 1),
                        48: (64, 1),
                        96: (78, 1),
                        192: (100, 1),
                        384: (128, 1),
                        # Needs int64
                        768: (78, 2),
                        1536: (100, 2),
                        3072: (128, 2),
                        6144: (78, 3),
                        12288: (100, 3),
                        24576: (128, 3),
                        49152: (156, 3)}
elif degree == 4:
    # Degree 4 mesh sizes
    target_mesh_size = {24: (17, 1),
                        48: (22, 1),
                        96: (27, 1),
                        192: (34, 1),
                        384: (44, 1),
                        768: (54, 1),
                        # Needs int64
                        1536: (68, 1),
                        3072: (88, 1),
                        6144: (108, 1),
                        12288: (68, 2),
                        24576: (88, 2),
                        49152: (108, 2)}

run_solve(problem_cls, degree, target_mesh_size[COMM_WORLD.size])
