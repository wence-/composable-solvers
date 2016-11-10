# flake8: noqa: F403
from firedrake import *
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage

import numpy
from argparse import ArgumentParser


parser = ArgumentParser(description="""Set options for temperature-driven Rayleigh-Benard.  Uses Taylor-Hood elements""",
                        add_help=False)

parser.add_argument("--degree", action="store", default=1,
                    help="Polynomial degree of the temperature and pressure spaces",
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

parser.add_argument("--output_solution", action="store_true",
                    help="Output the solution for visualisation in paraview")

parser.add_argument("--help", action="store_true",
                    help="Show help")

parser.add_argument("--vertical-temperature", action="store_true",
                    help="Apply a vertical temperature gradient (default horizontal)")

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
        self.Ra = Constant(args.Ra)
        self.Pr = Constant(args.Pr)
        self.args = args

    @cached_property
    def function_space(self):
        if self.dimension == 2:
            mesh = RectangleMesh(self.N, self.N, 3, 1)
        elif self.dimension == 3:
            mesh = UnitCubeMesh(self.N, self.N, self.N)
        else:
            raise ValueError("Invalid dimension, %d", self.dimension)

        V = VectorFunctionSpace(mesh, "CG", self.degree+1)
        P = FunctionSpace(mesh, "CG", self.degree)
        T = FunctionSpace(mesh, "CG", self.degree)
        return V*P*T

    @cached_property
    def w(self):
        return Function(self.function_space, name="solution")

    @cached_property
    def F(self):
        W = self.function_space

        u, p, T = split(self.w)
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
            if self.args.vertical_temperature:
                high_T = 3      # bottom
                low_T = 4       # top
            else:
                high_T = 1      # left
                low_T = 2       # right
        else:
            if self.args.vertical_temperature:
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
    def solver(self):
        problem = NonlinearVariationalProblem(self.F, self.w,
                                              bcs=self.bcs)
        solver = NonlinearVariationalSolver(problem, options_prefix="",
                                            nullspace=self.nullspace,
                                            solver_parameters={"mat_type": "matfree"},
                                            appctx={"velocity_space": 0})
        return solver

    def output(self):
        if args.output_solution:
            u, p, T = self.w.split()
            u.rename("Velocity")
            p.rename("Pressure")
            T.rename("Temperature")
            File("solution.pvd").write(u, p, T)


def run():
    with timed_stage("Problem setup"):
        problem = Problem(args)
        solver = problem.solver()

    with timed_stage("Solve problem"):
        solver.solve()

    with timed_stage("Output solution"):
        problem.output()


run()


