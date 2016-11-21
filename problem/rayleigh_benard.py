# flake8: noqa: F403
from __future__ import absolute_import
from argparse import ArgumentParser
from firedrake import *
from firedrake.utils import cached_property

from . import baseproblem


class Problem(baseproblem.Problem):

    name = "Rayleigh-Benard"

    def __init__(self):
        super(Problem, self).__init__()
        args, _ = self.argparser.parse_known_args()
        if args.help:
            self.argparser.print_help()
            import sys
            sys.exit(0)
        self.degree = args.degree
        self.dimension = args.dimension
        self.N = args.size
        self.Ra = Constant(args.Ra)
        self.Pr = Constant(args.Pr)
        self.vertical_temperature = args.vertical_temperature

    @cached_property
    def argparser(self):
        parser = ArgumentParser(description="""Set options for driven-cavity Navier-Stokes.  Uses Taylor-Hood elements""",
                                add_help=False)

        parser.add_argument("--degree", action="store", default=1,
                            help="Polynomial degree of the pressure space",
                            type=int)

        parser.add_argument("--size", action="store",  default=10,
                            help="Number of cells in each spatial direction",
                            type=int)

        parser.add_argument("--dimension", action="store", default=2, choices=[2, 3],
                            help="Spatial dimension of problem",
                            type=int)

        parser.add_argument("--Ra", action="store", default=200,
                            help="Rayleigh number",
                            type=float)

        parser.add_argument("--Pr", action="store", default=6.8,
                            help="Prandtl number",
                            type=float)

        parser.add_argument("--vertical-temperature", action="store_true",
                            default=False,
                            help="Apply a vertical temperature gradient?")

        parser.add_argument("--help", action="store_true",
                            help="Show help")

        return parser

    @cached_property
    def function_space(self):
        V = VectorFunctionSpace(self.mesh, "CG", self.degree+1)
        P = FunctionSpace(self.mesh, "CG", self.degree)
        T = FunctionSpace(self.mesh, "CG", self.degree)
        return V*P*T

    @cached_property
    def F(self):
        W = self.function_space

        u, p, T = split(self.u)
        v, q, S = TestFunctions(W)

        if self.dimension == 2:
            g = Constant((0, -1))
        else:
            g = Constant((0, 0, -1))

        F = (
            inner(grad(u), grad(v))*dx
            + inner(dot(grad(u), u), v)*dx
            - inner(p, div(v))*dx
            - self.Ra*self.Pr*inner(T*g, v)*dx
            + inner(div(u), q)*dx
            + inner(dot(grad(T), u), S)*dx
            + 1/self.Pr * inner(grad(T), grad(S))*dx
            )
        return F

    @cached_property
    def bcs(self):
        if self.dimension == 2:
            if self.vertical_temperature:
                high_T = 3      # bottom
                low_T = 4       # top
            else:
                high_T = 1      # left
                low_T = 2       # right
        else:
            if self.vertical_temperature:
                high_T = 5      # bottom
                low_T = 6       # top
            else:
                high_T = 1      # left
                low_T = 2       # right
        return (DirichletBC(self.function_space.sub(0), zero(self.dimension), "on_boundary"),
                DirichletBC(self.function_space.sub(2), Constant(1.0), high_T),
                DirichletBC(self.function_space.sub(2), Constant(0.0), low_T))

    @cached_property
    def nullspace(self):
        return MixedVectorSpaceBasis(self.function_space,
                                     [self.function_space.sub(0),
                                      VectorSpaceBasis(constant=True),
                                      self.function_space.sub(2)])

    @cached_property
    def appctx(self):
        return {"velocity_space": 0}

    @cached_property
    def output_fields(self):
        u, p, T = self.u.split()
        u.rename("Velocity")
        p.rename("Pressure")
        T.rename("Temperature")
        return (u, p, T)
