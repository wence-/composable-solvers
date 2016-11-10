# flake8: noqa: F403
from firedrake import *
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage

import numpy
from argparse import ArgumentParser


set_log_level(DEBUG)
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

parser.add_argument("--Re", action="store", default=10,
                    help="Reynolds number",
                    type=float)

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
        self.Re = Constant(args.Re)

    @cached_property
    def function_space(self):
        if self.dimension == 2:
            mesh = UnitSquareMesh(self.N, self.N)
        elif self.dimension == 3:
            mesh = UnitCubeMesh(self.N, self.N, self.N)
        else:
            raise ValueError("Invalid dimension, %d", self.dimension)

        V = VectorFunctionSpace(mesh, "CG", self.degree+1)
        P = FunctionSpace(mesh, "CG", self.degree)
        return V*P

    @cached_property
    def w(self):
        return Function(self.function_space, name="solution")

    @cached_property
    def F(self):
        W = self.function_space

        u, p = split(self.w)
        v, q = TestFunctions(W)

        F = ((1.0/self.Re) * inner(grad(u), grad(v))*dx
             + inner(dot(grad(u), u), v)*dx
             - p*div(v)*dx
             + div(u)*q*dx)

        return F

    @cached_property
    def bcs(self):
        if self.dimension == 2:
            top = 4
            rest = (1, 2, 3)
            drive = Constant((1, 0))
        else:
            top = 6
            rest = (1, 2, 3, 4, 5)
            drive = Constant((1, 0, 0))
        return (DirichletBC(self.function_space.sub(0), drive, top),
                DirichletBC(self.function_space.sub(0), zero(self.dimension), rest))

    @cached_property
    def nullspace(self):
        return MixedVectorSpaceBasis(self.function_space,
                                     [self.function_space.sub(0),
                                      VectorSpaceBasis(constant=True)])
    def solver(self):
        problem = NonlinearVariationalProblem(self.F, self.w,
                                              bcs=self.bcs)
        solver = NonlinearVariationalSolver(problem, options_prefix="",
                                            nullspace=self.nullspace,
                                            solver_parameters={"mat_type": "matfree"},
                                            appctx={"velocity_space": 0,
                                                    "Re": self.Re})
        return solver

    def output(self):
        if args.output_solution:
            u, p = self.w.split()
            u.rename("Velocity")
            p.rename("Pressure")
            File("solution.pvd").write(u, p)


def run():
    with timed_stage("Problem setup"):
        problem = Problem(args)
        solver = problem.solver()

    with timed_stage("Solve problem"):
        solver.solve()

    with timed_stage("Output solution"):
        problem.output()


run()


