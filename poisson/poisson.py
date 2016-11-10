# flake8: noqa: F403
from firedrake import *
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage

import numpy
from argparse import ArgumentParser


parser = ArgumentParser(description="""Set options for Poisson problem""", add_help=False)

parser.add_argument("--degree", action="store", default=1,
                    help="Polynomial degree",
                    type=int)

parser.add_argument("--size", action="store",  default=10,
                    help="Number of cells in each spatial direction",
                    type=int)

parser.add_argument("--dimension", action="store", default=2, choices=[1, 2, 3],
                    help="Spatial dimension of problem",
                    type=int)

parser.add_argument("--random", action="store_true",
                    help="Use a random right hand side (otherwise use one with an exact solution)")

parser.add_argument("--output_solution", action="store_true",
                    help="Output the solution for visualisation in paraview")

parser.add_argument("--help", action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.help:
    parser.print_help()
    import sys
    sys.exit(0)

class Problem(object):
    def __init__(self, args):
        self.degree = args.degree
        self.dimension = args.dimension
        self.N = args.size
        self.random = args.random
        self.args = args

    @cached_property
    def function_space(self):
        if self.dimension == 1:
            mesh = UnitIntervalMesh(self.N)
        elif self.dimension == 2:
            mesh = UnitSquareMesh(self.N, self.N)
        elif self.dimension == 3:
            mesh = UnitCubeMesh(self.N, self.N, self.N)
        else:
            raise ValueError("Invalid dimension, %d", self.dimension)

        return FunctionSpace(mesh, "CG", self.degree)

    @cached_property
    def u(self):
        return Function(self.function_space, name="solution")

    @cached_property
    def forcing(self):
        V = self.function_space
        if self.random:
            f = Function(V)
            numpy.random.seed(f.comm.rank + 101)
            f.dat.data[:] = numpy.random.exponential(scale=10, size=f.dat.data_ro.shape)
        else:
            f = Constant(1)

        return f

    @cached_property
    def a(self):
        V = self.function_space
        v = TestFunction(V)
        return derivative(inner(grad(self.u), grad(v))*dx, self.u)

    @cached_property
    def L(self):
        v = TestFunction(self.function_space)
        return self.forcing*v*dx

    @cached_property
    def bcs(self):
        return DirichletBC(self.function_space, 0, "on_boundary")

    def solver(self):
        problem = LinearVariationalProblem(self.a, self.L, self.u,
                                           bcs=self.bcs)
        solver = LinearVariationalSolver(problem, options_prefix="")
        return solver

    def output(self):
        if args.output_solution:
            File("solution.pvd").write(self.u)


def run():
    with timed_stage("Problem setup"):
        problem = Problem(args)

        solver = problem.solver()

    with timed_stage("Solve problem"):
        solver.solve()

    with timed_stage("Output solution"):
        problem.output()

run()
