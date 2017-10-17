# flake8: noqa: F403
from __future__ import absolute_import
from argparse import ArgumentParser
from firedrake import *
from firedrake.utils import cached_property

from . import baseproblem


class Problem(baseproblem.Problem):

    name = "Rayleigh-Benard"

    parameter_names = ("mumps", "pcd_mg")

    @property
    def mumps(self):
        return {"snes_type": "newtonls",
                "snes_monitor": True,
                "snes_converged_reason": True,
                "ksp_converged_reason": True,
                "snes_rtol": 1e-8,
                "snes_linesearch_type": "basic",
                "ksp_type": "preonly",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_package": "mumps"}

    @property
    def hypre(self):
        return {"pc_hypre_type": "boomeramg",
                "pc_hypre_boomeramg_no_CF": True,
                "pc_hypre_boomeramg_coarsen_type": "HMIS",
                "pc_hypre_boomeramg_interp_type": "ext+i",
                "pc_hypre_boomeramg_P_max": 4,
                "pc_hypre_boomeramg_agg_nl": 1,
                "pc_hypre_boomeramg_agg_num_paths": 2}
    @property
    def pcd_mg(self):
        pcd_mg = {"snes_type": "newtonls",
                  "snes_view": True,
                  "snes_monitor": True,
                  "snes_rtol": 1e-8,
                  "snes_converged_reason": True,
                  "ksp_converged_reason": True,
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
                  "fieldsplit_0_ksp_converged_reason": True,
                  "fieldsplit_0_ksp_gmres_modifiedgramschmidt": True,
                  "fieldsplit_0_ksp_rtol": 1e-2,
                  "fieldsplit_0_pc_type": "fieldsplit",
                  "fieldsplit_0_pc_fieldsplit_type": "schur",
                  "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
                  # HYPRE on velocity block
                  "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
                  "fieldsplit_0_fieldsplit_0_pc_type": "python",
                  "fieldsplit_0_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                  "fieldsplit_0_fieldsplit_0_assembled_mat_type": "aij",
                  "fieldsplit_0_fieldsplit_0_assembled_pc_type": "hypre",
                  # PCD on the pressure block
                  "fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
                  "fieldsplit_0_fieldsplit_1_pc_type": "python",
                  "fieldsplit_0_fieldsplit_1_pc_python_type": "firedrake.PCDPC",
                  # Matrix-free Fp application
                  "fieldsplit_0_fieldsplit_1_pcd_Fp_mat_type": "matfree",
                  # sor on assembled mass matrix
                  "fieldsplit_0_fieldsplit_1_pcd_Mp_mat_type": "aij",
                  "fieldsplit_0_fieldsplit_1_pcd_Mp_ksp_type": "richardson",
                  "fieldsplit_0_fieldsplit_1_pcd_Mp_ksp_max_it": 2,
                  "fieldsplit_0_fieldsplit_1_pcd_Mp_pc_type": "sor",
                  # hypre on assembled stiffness matrix
                  "fieldsplit_0_fieldsplit_1_pcd_Kp_ksp_type": "preonly",
                  "fieldsplit_0_fieldsplit_1_pcd_Kp_mat_type": "aij",
                  "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_type": "telescope",
                  "fieldsplit_0_fieldsplit_1_pcd_Kp_pc_telescope_reduction_factor": 6,
                  "fieldsplit_0_fieldsplit_1_pcd_Kp_telescope_pc_type": "hypre",
                  # hypre on temperature block
                  "fieldsplit_1_ksp_type": "gmres",
                  "fieldsplit_1_ksp_converged_reason": True,
                  "fieldsplit_1_ksp_rtol": 1e-4,
                  "fieldsplit_1_pc_type": "python",
                  "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
                  "fieldsplit_1_assembled_mat_type": "aij",
                  "fieldsplit_1_assembled_pc_type": "telescope",
                  "fieldsplit_1_assembled_pc_telescope_reduction_factor": 6,
                  "fieldsplit_1_assembled_telescope_pc_type": "hypre"}
        for k, v in self.hypre.items():
            if k.startswith("pc_hypre_boomeramg"):
                pcd_mg["fieldsplit_1_assembled_telescope_%s" % k] = v
                pcd_mg["fieldsplit_0_fieldsplit_1_pcd_Kp_telescope_%s" % k] = v
                pcd_mg["fieldsplit_0_fieldsplit_0_assembled_%s" % k] = v
        return pcd_mg

    @cached_property
    def Ra(self):
        return Constant(self.args.Ra)

    @cached_property
    def Pr(self):
        return Constant(self.args.Pr)

    @property
    def vertical_temperature(self):
        return self.args.vertical_temperature

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
            + (self.Ra/self.Pr)*inner(T*g, v)*dx
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
