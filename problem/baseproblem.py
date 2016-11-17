# flake8: noqa: F403
from firedrake import *
from firedrake.utils import cached_property
from abc import ABCMeta, abstractproperty, abstractmethod


class Problem(object):
    __metaclass__ = ABCMeta

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

    def solver(self):
        problem = NonlinearVariationalProblem(self.F, self.u, bcs=self.bcs,
                                              Jp=self.Jp)
        solver = NonlinearVariationalSolver(problem, options_prefix="",
                                            nullspace=self.nullspace,
                                            near_nullspace=self.near_nullspace,
                                            appctx=self.appctx)
        return solver

    @abstractproperty
    def argparser(self):
        pass

    @abstractproperty
    def output_fields(self):
        pass

    @abstractproperty
    def datastore_name(self):
        pass
