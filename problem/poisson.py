# flake8: noqa: F403
from __future__ import absolute_import
from argparse import ArgumentParser
from firedrake import *
from firedrake.utils import cached_property
import numpy

from . import baseproblem


class Problem(baseproblem.Problem):

    name = "Poisson"

    def __init__(self):
        super(Problem, self).__init__()
        args, _ = self.argparser.parse_known_args()
        if args.help:
            import sys
            self.argparser.print_help()
            sys.exit(0)
        self.degree = args.degree
        self.dimension = args.dimension
        self.N = args.size
        self.random = args.random
        self.args = args

    @cached_property
    def argparser(self):
        parser = ArgumentParser(description="""Set options for Poisson problem""", add_help=False)

        parser.add_argument("--degree", action="store", default=1,
                            help="Polynomial degree",
                            type=int)

        parser.add_argument("--size", action="store",  default=10,
                            help="Number of cells in each spatial direction",
                            type=int)

        parser.add_argument("--dimension", action="store", default=2, choices=[2, 3],
                            help="Spatial dimension of problem",
                            type=int)

        parser.add_argument("--random", action="store_true",
                            help="Use a random right hand side (otherwise use one with an exact solution)")

        parser.add_argument("--output_solution", action="store_true",
                            help="Output the solution for visualisation in paraview")

        parser.add_argument("--help", action="store_true",
                            help="Show help")
        return parser

    @cached_property
    def function_space(self):
        return FunctionSpace(self.mesh, "CG", self.degree)

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
    def F(self):
        V = self.function_space
        v = TestFunction(V)
        return inner(grad(self.u), grad(v))*dx - self.forcing*v*dx

    @cached_property
    def bcs(self):
        return DirichletBC(self.function_space, 0, "on_boundary")

    @cached_property
    def output_fields(self):
        return (self.u, )
