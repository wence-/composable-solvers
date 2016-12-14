# flake8: noqa: F403
from __future__ import absolute_import
from argparse import ArgumentParser
from firedrake import *
from firedrake.utils import cached_property

from . import baseproblem


class Problem(baseproblem.Problem):

    name = "Elasticity"

    parameter_names = ("mumps", "hypre", "gamg", "schwarz", "schwarzmf")

    hypre = {"snes_type": "ksponly",
             "ksp_type": "cg",
             "ksp_rtol": 1e-8,
             "ksp_monitor": True,
             "pc_type": "hypre",
             "pc_hypre_type": "boomeramg",
             "mat_type": "aij"}

    gamg = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1e-8,
            "ksp_monitor": True,
            "pc_type": "gamg",
            "mat_type": "aij"}

    mumps = {"snes_type": "ksponly",
             "ksp_type": "preonly",
             "pc_type": "lu",
             "pc_factor_mat_solver_package": "mumps",
             "mat_type": "baij"}

    schwarz = {"snes_type": "ksponly",
               "ksp_type": "cg",
               "ksp_rtol": 1e-8,
               "ksp_monitor": True,
               "mat_type": "matfree",
               "pc_type": "python",
               "pc_python_type": "ssc.SSC",
               "ssc_pc_composite_type": "additive",
               # Patch config
               "ssc_sub_0_pc_patch_save_operators": True,
               "ssc_sub_0_pc_patch_sub_mat_type": "seqaij",
               "ssc_sub_0_sub_ksp_type": "preonly",
               "ssc_sub_0_sub_pc_type": "lu",
               # Low-order config
               "ssc_sub_1_lo_pc_type": "gamg",
               "ssc_sub_1_lo_mat_type": "aij",}

    schwarzmf = {"snes_type": "ksponly",
                 "ksp_type": "cg",
                 "ksp_rtol": 1e-8,
                 "ksp_monitor": True,
                 "mat_type": "matfree",
                 "pc_type": "python",
                 "pc_python_type": "ssc.SSC",
                 "ssc_pc_composite_type": "additive",
                 # Patch config
                 "ssc_sub_0_pc_patch_save_operators": False,
                 "ssc_sub_0_pc_patch_sub_mat_type": "seqaij",
                 "ssc_sub_0_sub_ksp_type": "preonly",
                 "ssc_sub_0_sub_pc_type": "lu",
                 # Low-order config
                 "ssc_sub_1_lo_pc_type": "gamg",
                 "ssc_sub_1_lo_mat_type": "aij",}

    @property
    def nu(self):
        return self.args.nu

    @property
    def lmbda(self):
        return self.args.lmbda

    @staticmethod
    def argparser():
        parser = ArgumentParser(description="""Set options for Elasticity problem""", add_help=False)

        parser.add_argument("--degree", action="store", default=1,
                            help="Polynomial degree",
                            type=int)

        parser.add_argument("--size", action="store",  default=10,
                            help="Number of cells in each spatial direction",
                            type=int)

        parser.add_argument("--dimension", action="store", default=2, choices=[2, 3],
                            help="Spatial dimension of problem",
                            type=int)

        parser.add_argument("--nu", action="store", default=0.3,
                            help="Poisson ratio", type=float)

        parser.add_argument("--lmbda", action="store", default=6,
                            help="First Lame parameter", type=float)

        parser.add_argument("--output_solution", action="store_true",
                            help="Output the solution for visualisation in paraview")

        parser.add_argument("--help", action="store_true",
                            help="Show help")
        return parser

    @cached_property
    def function_space(self):
        return VectorFunctionSpace(self.mesh, "CG", self.degree)

    @cached_property
    def mu(self):
        return self.lmbda*(1 - 2*self.nu)/(2*self.nu)

    @cached_property
    def forcing(self):
        # Squash under gravity
        if self.dimension == 2:
            f = Constant((0, -10))
        else:
            f = Constant((0, 0, -10))
        return f

    def sigma(self, v):
        return 2.0*self.mu*sym(grad(v)) + self.lmbda*tr(sym(grad(v)))*Identity(self.dimension)

    @cached_property
    def F(self):
        V = self.function_space
        v = TestFunction(V)
        return inner(self.sigma(self.u), grad(v))*dx - inner(self.forcing, v)*dx

    @cached_property
    def bcs(self):
        # Could add some space-dependent rotation/shear
        if self.dimension == 2:
            return (DirichletBC(self.function_space, Constant((-0.1, 0)), 3),
                    DirichletBC(self.function_space, Constant((0.1, 0)), 4))
        else:
            return (DirichletBC(self.function_space, Constant((-0.1, 0, 0)), 5),
                    DirichletBC(self.function_space, Constant((0.1, 0, 0)), 6))

    @cached_property
    def near_nullspace(self):
        if self.dimension == 2:
            basis = tuple(Function(self.function_space) for _ in range(3))
            x, y = SpatialCoordinate(self.function_space.ufl_domain())
            basis[0].interpolate(Constant((1, 0)))
            basis[1].interpolate(Constant((0, 1)))
            basis[2].interpolate(as_vector([-y, x]))
        else:
            basis = tuple(Function(self.function_space) for _ in range(6))
            x, y, z = SpatialCoordinate(self.function_space.ufl_domain())
            # Translational modes
            basis[0].interpolate(Constant((1, 0, 0)))
            basis[1].interpolate(Constant((0, 1, 0)))
            basis[2].interpolate(Constant((0, 0, 1)))

            # Rotational modes
            basis[3].interpolate(as_vector([-y, x, 0]))
            basis[4].interpolate(as_vector([z, 0, -x]))
            basis[5].interpolate(as_vector([0, -z, y]))

        for i, b in enumerate(basis):
            alphas = []
            for b_ in basis[:i]:
                alphas.append(b.dat.inner(b_.dat))
            for alpha, b_ in zip(alphas, basis[:i]):
                b.dat -= b_.dat * alpha
            b.dat /= b.dat.norm
        return VectorSpaceBasis(vecs=basis)

    @property
    def output_fields(self):
        stress = Function(TensorFunctionSpace(self.function_space.ufl_domain(),
                                              "DG", 1),
                          name="stress")
        stress.interpolate(self.sigma(self.u))
        return (self.u, stress)
