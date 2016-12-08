# flake8: noqa: F403
from __future__ import absolute_import
from argparse import ArgumentParser
from firedrake import *
from firedrake.utils import cached_property

from . import baseproblem


class Problem(baseproblem.Problem):

    name = "Navier-Stokes"

    parameter_names = ("mumps", "pcd_lu", "pcd_mg", "pcd_schwarz")[-1:]

    mumps = {"snes_type": "newtonls",
             "snes_monitor": True,
             "snes_rtol": 1e-8,
             "snes_linesearch_type": "basic",
             "ksp_type": "preonly",
             "mat_type": "aij",
             "pc_type": "lu",
             "pc_factor_mat_solver_package": "mumps"}

    pcd_lu = {"snes_type": "newtonls",
              "snes_monitor": True,
              "snes_rtol": 1e-8,
              "snes_linesearch_type": "basic",
              "ksp_type": "fgmres",
              "ksp_monitor": True,
              "ksp_gmres_modifiedgramschmidt": True,
              "mat_type": "matfree",
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "schur",
              "pc_fieldsplit_schur_fact_type": "lower",
              # Mumps on velocity block
              "fieldsplit_0_ksp_type": "preonly",
              "fieldsplit_0_pc_type": "python",
              "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_0_assembled_pc_type": "lu",
              "fieldsplit_0_assembled_pc_factor_mat_solver_package": "mumps",
              # PCD on pressure block
              "fieldsplit_1_ksp_type": "gmres",
              "fieldsplit_1_ksp_rtol": 1e-4,
              "fieldsplit_1_pc_type": "python",
              "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
              "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
              "fieldsplit_1_pcd_Mp_pc_type": "lu",
              "fieldsplit_1_pcd_Mp_pc_factor_mat_solver_package": "mumps",
              "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
              "fieldsplit_1_pcd_Kp_pc_type": "lu",
              "fieldsplit_1_pcd_Kp_pc_factor_mat_solver_package": "mumps"}

    pcd_mg = {"snes_type": "newtonls",
              "snes_monitor": True,
              "snes_rtol": 1e-8,
              "snes_linesearch_type": "basic",
              "snes_ksp_ew": True,
              "ksp_type": "fgmres",
              "ksp_monitor": True,
              "ksp_gmres_modifiedgramschmidt": True,
              "mat_type": "matfree",
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "schur",
              "pc_fieldsplit_schur_fact_type": "lower",
              # hypre on velocity block
              "fieldsplit_0_ksp_type": "gmres",
              "fieldsplit_0_ksp_rtol": 1e-2,
              "fieldsplit_0_pc_type": "python",
              "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_0_assembled_pc_type": "hypre",
              "fieldsplit_0_assembled_mat_type": "aij",
              "fieldsplit_1_inner_ksp_type": "preonly",
              "fieldsplit_1_inner_pc_type": "python",
              "fieldsplit_1_inner_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_1_inner_assembled_pc_type": "hypre",
              "fieldsplit_1_inner_assembled_mat_type": "aij",
              # PCD on pressure block
              "fieldsplit_1_ksp_type": "gmres",
              "fieldsplit_1_ksp_rtol": 1e-2,
              "fieldsplit_1_pc_type": "python",
              "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
              "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
              "fieldsplit_1_pcd_Mp_pc_type": "sor",
              "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
              "fieldsplit_1_pcd_Kp_pc_type": "hypre"}

    pcd_schwarz = {"snes_type": "newtonls",
                   "snes_monitor": True,
                   "snes_lag_preconditioner": 2,
                   "snes_rtol": 1e-8,
                   "snes_linesearch_type": "basic",
                   "snes_ksp_ew": True,
                   "ksp_type": "fgmres",
                   "ksp_monitor": True,
                   "ksp_gmres_modifiedgramschmidt": True,
                   "mat_type": "matfree",
                   "pc_type": "fieldsplit",
                   "pc_fieldsplit_type": "schur",
                   "pc_fieldsplit_schur_fact_type": "lower",
                   # Schwarz on velocity block
                   "fieldsplit_0_ksp_type": "preonly",
                   "fieldsplit_0_pc_type": "python",
                   "fieldsplit_0_pc_python_type": "ssc.SSC",
                   "fieldsplit_0_pc_composite_type": "additive",
                   "fieldsplit_0_ssc_sub_0_pc_patch_save_operators": True,
                   "fieldsplit_0_ssc_sub_0_pc_patch_sub_mat_type": "seqdense",
                   "fieldsplit_0_ssc_sub_0_sub_ksp_type": "preonly",
                   "fieldsplit_0_ssc_sub_0_sub_pc_type": "lu",
                   "fieldsplit_0_ssc_sub_1_lo_pc_type": "hypre",
                   "fieldsplit_0_ssc_sub_1_lo_mat_type": "aij",
                   # PCD on pressure block
                   "fieldsplit_1_ksp_type": "gmres",
                   "fieldsplit_1_ksp_rtol": 1e-2,
                   "fieldsplit_1_pc_type": "python",
                   "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
                   "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
                   "fieldsplit_1_pcd_Mp_pc_type": "sor",
                   "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
                   "fieldsplit_1_pcd_Kp_pc_type": "hypre"}

    def __init__(self, N=None, degree=None, dimension=None):
        super(Problem, self).__init__(N, degree, dimension)
        self.Re = Constant(self.args.Re)

    @staticmethod
    def argparser():
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

        return parser

    @cached_property
    def function_space(self):
        V = VectorFunctionSpace(self.mesh, "CG", self.degree+1)
        P = FunctionSpace(self.mesh, "CG", self.degree)
        return V*P

    @cached_property
    def F(self):
        W = self.function_space

        u, p = split(self.u)
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
    def appctx(self):
        return {"velocity_space": 0,
                "Re": self.Re}

    @cached_property
    def nullspace(self):
        return MixedVectorSpaceBasis(self.function_space,
                                     [self.function_space.sub(0),
                                      VectorSpaceBasis(constant=True)])

    @cached_property
    def output_fields(self):
        u, p = self.u.split()
        u.rename("Velocity")
        p.rename("Pressure")
        return (u, p)
