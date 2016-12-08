# flake8: noqa: F403
from __future__ import absolute_import
from argparse import ArgumentParser
from firedrake import *
from firedrake.utils import cached_property

from . import baseproblem


class Problem(baseproblem.Problem):

    name = "Rayleigh-Benard"

    parameter_names = ("mumps", "pcd_lu", "pcd_mg", "pcd_schwarz_everywhere")

    mumps = {"snes_type": "newtonls",
             "snes_monitor": True,
             "snes_rtol": 1e-8,
             "snes_linesearch_type": "basic",
             "ksp_type": "preonly",
             "mat_type": "aij",
             "pc_type": "lu",
             "pc_factor_mat_solver_package": "mumps"}

    # Zero pivots sometimes occur?
    pcd_lu = {"snes_type": "newtonls",
              "snes_monitor": True,
              "snes_rtol": 1e-8,
              "snes_linesearch_type": "basic",
              "mat_type": "matfree",
              "ksp_type": "fgmres",
              "ksp_monitor": True,
              "ksp_gmres_modifiedgramschmidt": True,
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "multiplicative",
              "pc_fieldsplit_0_fields": "0,1",
              "pc_fieldsplit_1_fields": "2",
              # GMRES on Navier-stokes, with fieldsplit PC.
              "fieldsplit_0_ksp_type": "gmres",
              "fieldsplit_0_ksp_gmres_modifiedgramschmidt": True,
              "fieldsplit_0_ksp_rtol": 1e-2,
              "fieldsplit_0_pc_type": "fieldsplit",
              "fieldsplit_0_pc_fieldsplit_type": "schur",
              "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
              # LU on velocity block
              "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
              "fieldsplit_0_fieldsplit_0_pc_type": "python",
              "fieldsplit_0_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_0_fieldsplit_0_assembled_pc_type": "lu",
              "fieldsplit_0_fieldsplit_0_assembled_mat_type": "baij",
              "fieldsplit_0_fieldsplit_0_assembled_pc_factor_mat_solver_package": "mumps",
              # PCD on the pressure block
              "fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
              "fieldsplit_0_fieldsplit_1_pc_type": "python",
              "fieldsplit_0_fieldsplit_1_pc_python_type": "firedrake.PCDPC",
              # Matrix-free Fp application
              "fieldsplit_0_fieldsplit_1_pcd_Fp_mat_type": "matfree",
              # lu on assembled mass matrix
              "fieldsplit_0_fieldsplit_1_pcd_Mp_ksp_type": "preonly",
              "fieldsplit_0_fieldsplit_1_pcd_Mp_mat_type": "aij",
              "fieldsplit_0_fieldsplit_1_pcd_Mp_pc_type": "lu",
              "fieldsplit_0_fieldsplit_1_pcd_Mp_pc_factor_mat_solver_package": "mumps",
              # lu on assembled stiffness matrix
              "fieldsplit_0_fieldsplit_1_pcd_Kp_ksp_type": "preonly",
              "fieldsplit_0_fieldsplit_1_pcd_Kp_mat_type": "aij",
              "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_type": "lu",
              "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_factor_mat_solver_package": "mumps",
              # LU on temperature block
              "fieldsplit_1_ksp_type": "preonly",
              "fieldsplit_1_pc_type": "python",
              "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_1_assembled_pc_type": "lu",
              "fieldsplit_1_assembled_pc_factor_mat_solver_package": "mumps"}

    pcd_mg = {"snes_type": "newtonls",
              "snes_monitor": True,
              "snes_rtol": 1e-8,
              "snes_linesearch_type": "basic",
              "mat_type": "matfree",
              "ksp_type": "fgmres",
              "ksp_monitor": True,
              "ksp_gmres_modifiedgramschmidt": True,
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "multiplicative",
              "pc_fieldsplit_0_fields": "0,1",
              "pc_fieldsplit_1_fields": "2",
              # GMRES on Navier-stokes, with fieldsplit PC.
              "fieldsplit_0_ksp_type": "gmres",
              "fieldsplit_0_ksp_gmres_modifiedgramschmidt": True,
              "fieldsplit_0_ksp_rtol": 1e-2,
              "fieldsplit_0_pc_type": "fieldsplit",
              "fieldsplit_0_pc_fieldsplit_type": "schur",
              "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
              # GAMG on velocity block
              "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
              "fieldsplit_0_fieldsplit_0_pc_type": "python",
              "fieldsplit_0_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_0_fieldsplit_0_assembled_mat_type": "aij",
              "fieldsplit_0_fieldsplit_0_assembled_pc_type": "gamg",
              # PCD on the pressure block
              "fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
              "fieldsplit_0_fieldsplit_1_pc_type": "python",
              "fieldsplit_0_fieldsplit_1_pc_python_type": "firedrake.PCDPC",
              # Matrix-free Fp application
              "fieldsplit_0_fieldsplit_1_pcd_Fp_mat_type": "matfree",
              # sor on assembled mass matrix
              "fieldsplit_0_fieldsplit_1_pcd_Mp_mat_type": "aij",
              "fieldsplit_0_fieldsplit_1_pcd_Mp_ksp_type": "cg",
              "fieldsplit_0_fieldsplit_1_pcd_Mp_ksp_rtol": 1e-4,
              "fieldsplit_0_fieldsplit_1_pcd_Mp_pc_type": "sor",
              # gamg on assembled stiffness matrix
              "fieldsplit_0_fieldsplit_1_pcd_Kp_ksp_type": "cg",
              "fieldsplit_0_fieldsplit_1_pcd_Kp_ksp_rtol": 1e-4,
              "fieldsplit_0_fieldsplit_1_pcd_Kp_mat_type": "aij",
              "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_type": "gamg",
              # gamg on temperature block
              "fieldsplit_1_ksp_type": "gmres",
              "fieldsplit_1_ksp_rtol": 1e-2,
              "fieldsplit_1_pc_type": "python",
              "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_1_assembled_mat_type": "aij",
              "fieldsplit_1_assembled_pc_type": "gamg"}

    pcd_schwarz_everywhere = {"snes_type": "newtonls",
                              "snes_monitor": True,
                              "snes_rtol": 1e-8,
                              "snes_linesearch_type": "basic",
                              "mat_type": "matfree",
                              "ksp_type": "fgmres",
                              "ksp_monitor": True,
                              "ksp_gmres_modifiedgramschmidt": True,
                              "pc_type": "fieldsplit",
                              "pc_fieldsplit_type": "multiplicative",
                              "pc_fieldsplit_0_fields": "0,1",
                              "pc_fieldsplit_1_fields": "2",
                              # GMRES on Navier-stokes, with fieldsplit PC.
                              "fieldsplit_0_ksp_type": "gmres",
                              "fieldsplit_0_ksp_gmres_modifiedgramschmidt": True,
                              "fieldsplit_0_ksp_rtol": 1e-2,
                              "fieldsplit_0_pc_type": "fieldsplit",
                              "fieldsplit_0_pc_fieldsplit_type": "schur",
                              "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
                              # Schwarz on velocity block
                              "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
                              "fieldsplit_0_fieldsplit_0_pc_type": "python",
                              "fieldsplit_0_fieldsplit_0_pc_python_type": "ssc.SSC",
                              "fieldsplit_0_fieldsplit_0_pc_composite_type": "additive",
                              "fieldsplit_0_fieldsplit_0_ssc_sub_0_pc_patch_save_operators": True,
                              "fieldsplit_0_fieldsplit_0_ssc_sub_0_pc_patch_sub_mat_type": "seqdense",
                              "fieldsplit_0_fieldsplit_0_ssc_sub_0_sub_ksp_type": "preonly",
                              "fieldsplit_0_fieldsplit_0_ssc_sub_0_sub_pc_type": "lu",
                              "fieldsplit_0_fieldsplit_0_ssc_sub_1_lo_pc_type": "hypre",
                              "fieldsplit_0_fieldsplit_0_ssc_sub_1_lo_mat_type": "aij",
                              # PCD on the pressure block
                              "fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
                              "fieldsplit_0_fieldsplit_1_pc_type": "python",
                              "fieldsplit_0_fieldsplit_1_pc_python_type": "firedrake.PCDPC",
                              # Matrix-free Fp application
                              "fieldsplit_0_fieldsplit_1_pcd_Fp_mat_type": "matfree",
                              # SOR on assembled mass matrix
                              "fieldsplit_0_fieldsplit_1_pcd_Mp_ksp_type": "cg",
                              "fieldsplit_0_fieldsplit_1_pcd_Mp_ksp_rtol": 1e-4,
                              "fieldsplit_0_fieldsplit_1_pcd_Mp_pc_type": "sor",
                              # Schwarz on unassembled stiffness matrix
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ksp_type": "cg",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ksp_rtol": 1e-4,
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_mat_type": "matfree",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_type": "python",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_python_type": "ssc.SSC",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_composite_type": "additive",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ssc_sub_0_pc_patch_save_operators": True,
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ssc_sub_0_pc_patch_sub_mat_type": "seqdense",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ssc_sub_0_sub_ksp_type": "preonly",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ssc_sub_0_sub_pc_type": "lu",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ssc_sub_1_lo_pc_type": "hypre",
                              "fieldsplit_0_fieldsplit_1_pcd_Kp_ssc_sub_1_lo_mat_type": "aij",
                              # GMRES + Schwarz on the temperature block
                              "fieldsplit_1_ksp_type": "gmres",
                              "fieldsplit_1_ksp_rtol": 1e-2,
                              "fieldsplit_1_ksp_gmres_modifiedgramschmidt": True,
                              "fieldsplit_1_pc_type": "python",
                              "fieldsplit_1_pc_python_type": "ssc.SSC",
                              "fieldsplit_1_pc_composite_type": "additive",
                              "fieldsplit_1_ssc_sub_0_pc_patch_save_operators": True,
                              "fieldsplit_1_ssc_sub_0_pc_patch_sub_mat_type": "seqdense",
                              "fieldsplit_1_ssc_sub_0_sub_ksp_type": "preonly",
                              "fieldsplit_1_ssc_sub_0_sub_pc_type": "lu",
                              "fieldsplit_1_ssc_sub_1_lo_pc_type": "hypre",
                              "fieldsplit_1_ssc_sub_1_lo_mat_type": "aij"}

    def __init__(self, N=None, degree=None, dimension=None):
        super(Problem, self).__init__(N, degree, dimension)
        self.Ra = Constant(self.args.Ra)
        self.Pr = Constant(self.args.Pr)
        self.vertical_temperature = self.args.vertical_temperature

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
        global dx
        W = self.function_space

        u, p, T = split(self.u)
        v, q, S = TestFunctions(W)

        if self.dimension == 2:
            g = Constant((0, -1))
        else:
            g = Constant((0, 0, -1))

        dx = dx(degree=2*(self.degree))
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
