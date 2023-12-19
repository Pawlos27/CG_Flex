
import numpy as np
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import GPy
import matplotlib.pyplot as plt
import cgflex_project.Module_Dependencymaker._errortermmaker as _errortermmaker
import cgflex_project.Module_Dependencymaker._errortermmaker_collection as _errortermmaker_collection
import cgflex_project.Module_Dependencymaker._kernelcombinator as _kernelcombinator
from dataclasses import dataclass, field
from typing import Any
import cgflex_project.Module_Dependencymaker._functionmaker as _functionmaker
import cgflex_project.Module_Dependencymaker._functionmaker_extreme_values as _functionmaker_extreme_values
import cgflex_project.Module_Graphmaker.nodemaker as nodemaker

from typing import Any, List, Type, Tuple
import cgflex_project.Module_Dependencymaker._inputloader as _inputloader
import random
import cgflex_project.Module_Dependencymaker._dependencymaker_initial_value_distributions as _dependencymaker_initial_value_distributions






class Dependencies:
    def __init__(self, errorterm_model:_errortermmaker.Dependency_errorterm, function_model: _functionmaker.Dependency_functions ):
        self.function_model = function_model
        self.errorterm_model = errorterm_model

    def calculate_normalized_value(self,x_values: list):

        value_function = self.function_model.calculate_value(x_inputs=x_values)
        value_errorterm = self.errorterm_model.errorterm_distribution.calc_from_distribution(x_inputs=x_values)
        value_combined = value_function + value_errorterm
        return value_combined
    



class IDependency_setter(metaclass=ABCMeta):
    @abstractmethod
    def set_dependencies(self, node: nodemaker.Nodeobject) :
     """Interface Method"""


class Dependency_setter_total_random_distributions(IDependency_setter):

    def __init__(self, value_distributions: _dependencymaker_initial_value_distributions.IInitial_value_distribution_collection):
        self.value_distributions = value_distributions
    
    def set_dependencies(self, node: nodemaker.Nodeobject):
        distribution = self.value_distributions.get_distribution()
        return distribution

class Dependency_setter_default(IDependency_setter):

    def __init__(self,kernel_combination_maker: _kernelcombinator.IKernelcombinator, function_maker: _functionmaker.IFunction_maker, errorterm_maker:_errortermmaker.IErrorterm_maker  ):
        self.kernel_combination_maker = kernel_combination_maker
        self.function_maker = function_maker
        self.errorterm_maker = errorterm_maker


    def set_dependencies(self, node: nodemaker.Nodeobject, range_of_output: Tuple[float, float] ):
        if node.source == True: # because sources have no parents
            dimensions = 1
        else : 
            dimensions = len(node.parents)
        kernel = self.kernel_combination_maker.combinate_kernels(dimensions=dimensions)
        function_model = self.function_maker.make_functions(kernel=kernel, errorterm_tolerance =self.errorterm_maker.maximum_tolerance , range_of_output= range_of_output) # function model is a Dependency_functions Object

        errorterm_model = self.errorterm_maker.make_errorterm(function_model=function_model) # should be a dependency_errorterm_object
        dependency = Dependencies( function_model=function_model, errorterm_model=errorterm_model)
        return dependency







if __name__ == "__main__":
        pass

