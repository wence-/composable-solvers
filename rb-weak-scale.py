from argparse import ArgumentParser
import os
import cPickle
import pandas
from collections import defaultdict
import problem.rayleigh_benard as module

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


def run_solve(problem_cls, degree):
    target_mesh_size = {24: (20, 1),
                        48: (25, 1),
                        96: (32, 1),
                        192: (40, 1),
                        384: (50, 1),
                        768: (64, 1),
                        1536: (80, 1),
                        3072: (100, 1),
                        6144: (125, 1),
                        12288: (160, 1)}
    size, ref = target_mesh_size[COMM_WORLD.size]
    problem = problem_cls(degree=degree, N=size, refinements=ref)

    first = True

    for name in parameter_names:
        parameters = getattr(problem, name)
        solver = problem.solver(parameters=parameters)
        PETSc.Sys.Print("\nSolving with parameter set %s, %s, %s..." % (name, problem.N, problem.degree))
        if not warm[(name, degree)]:
            PETSc.Sys.Print("Warmup solve")
            problem.u.assign(0)
            with PETSc.Log.Stage("Warmup"):
                try:
                    solver.solve()
                except:
                    PETSc.Sys.Print("Unable to solve %s, %s, %s" % (name, problem.N, problem.degree))
                    PETSc.Sys.Print("************************************")
                    import traceback
                    PETSc.Sys.Print(*traceback.format_stack())
                    PETSc.Sys.Print("************************************")
                    raise
                warm[(name, degree)] = True

        problem.u.assign(0)

        PETSc.Sys.Print("Timed solve")
        solver.snes.setConvergenceHistory()
        solver.snes.ksp.setConvergenceHistory()
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

                num_cells = problem.comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)
                if COMM_WORLD.rank == 0:
                    if not os.path.exists(os.path.dirname(results)):
                        os.makedirs(os.path.dirname(results))

                    if args.overwrite:
                        if first:
                            mode = "w"
                            header = True
                        else:
                            mode = "a"
                            header = False
                        first = False
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
                            "mesh_size": size * (2**ref),
                            "num_cells": num_cells,
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
                PETSc.Sys.Print("************************************")
                import traceback
                PETSc.Sys.Print(*traceback.format_stack())
                PETSc.Sys.Print("************************************")
                raise
        PETSc.Sys.Print("Solving with parameter set %s, %s, %s...done" % (name, problem.N, problem.degree))


run_solve(problem_cls, 1)
