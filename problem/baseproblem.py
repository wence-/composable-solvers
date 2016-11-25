# flake8: noqa: F403
from firedrake import *
from firedrake.utils import cached_property
from abc import ABCMeta, abstractproperty, abstractmethod


class Problem(object):
    __metaclass__ = ABCMeta

    autorefine = False

    def __init__(self, N=None, degree=None, dimension=None):
        super(Problem, self).__init__()
        args, _ = self.argparser.parse_known_args()
        if args.help:
            import sys
            self.parser.print_help()
            sys.exit(0)
        self.degree = degree or args.degree
        self.dimension = dimension or args.dimension
        self.N = N or args.size
        self.args = args

    @abstractproperty
    def parameter_names(self):
        pass

    @property
    def comm(self):
        return self.mesh.comm

    @cached_property
    def mesh(self):
        if self.dimension == 2:
            mesh = UnitSquareMesh(self.N, self.N)
            # Refinements to give approximately same number of dofs
            # irrespective of degree
            refinements = {1: 4,
                           2: 3,
                           3: 3,
                           4: 2,
                           5: 2,
                           6: 2,
                           7: 1,
                           8: 0}
        elif self.dimension == 3:
            mesh = UnitCubeMesh(self.N, self.N, self.N)
            refinements = {1: 4,
                           2: 3,
                           3: 2,
                           4: 2,
                           5: 1,
                           6: 0}
        else:
            raise ValueError("Invalid dimension, %d", self.dimension)
        if self.autorefine:
            dm = mesh._plex
            from firedrake.mg.impl import filter_exterior_facet_labels
            for _ in range(refinements.get(self.degree, 0)):
                dm.setRefinementUniform(True)
                dm = dm.refine()
                dm.removeLabel("interior_facets")
                dm.removeLabel("op2_core")
                dm.removeLabel("op2_non_core")
                dm.removeLabel("op2_exec_halo")
                dm.removeLabel("op2_non_exec_halo")
                filter_exterior_facet_labels(dm)
            mesh = Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(),
                        distribute=False, reorder=True)

        return mesh

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def function_space(self):
        pass

    @cached_property
    def u(self):
        return Function(self.function_space, name="solution")

    @abstractproperty
    def F(self):
        pass

    @cached_property
    def J(self):
        return derivative(self.F, self.u)

    @cached_property
    def Jp(self):
        return None

    @cached_property
    def bcs(self):
        return None

    @cached_property
    def nullspace(self):
        return None

    @cached_property
    def near_nullspace(self):
        return None

    @cached_property
    def appctx(self):
        return None

    def solver(self, parameters=None):
        problem = NonlinearVariationalProblem(self.F, self.u, bcs=self.bcs,
                                              Jp=self.Jp)
        solver = NonlinearVariationalSolver(problem, options_prefix="",
                                            nullspace=self.nullspace,
                                            near_nullspace=self.near_nullspace,
                                            appctx=self.appctx,
                                            solver_parameters=parameters)
        return solver

    @abstractproperty
    def argparser(self):
        pass

    @abstractproperty
    def output_fields(self):
        pass
