
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
    """
    This class is a core property of the nodes in our model. Every node object except sources carry a dependency object.
    It represent the dependency of a node regarding its parent nodes. 
    It combines a function model and an error term model to represent this dependency.
    It allows for the calculation and summation of the values of these models based on input values of its parent nodes, resulting in the value of the node.

    Explanation Dependencies: 
    Dependencies represent the causal relation of the value of a node by its parent mode, this class represents the model of our Dependencies.
    Each node's value in the DAG is represented by a dependency function D(x) = F(x) + E(x), where x denotes the values of its parent nodes. 
    The function model is representing F(x) as a deterministic function. 
    The errorterm_model is representing E(x) , introducing a deviation variable.
    

    Args:
        function_model (_functionmaker.Dependency_functions): An instance representing the function model.
        errorterm_model (_errortermmaker_collection.IErrordistribution): An instance representing the error term model.

    """
    def __init__(self, errorterm_model: _errortermmaker_collection.IErrordistribution, function_model: _functionmaker.Dependency_functions ):
        self.function_model = function_model
        self.errorterm_model = errorterm_model

    def calculate_normalized_value(self,x_values: list):
        """
        Calculates value of a node the combined value from the function model and error term model based on input values.

        Args:
            x_values (list): The list of input values on which the calculations are based.

        Returns:
            The combined value from the function and error term models.
        """

        value_function = self.function_model.calculate_value(x_inputs=x_values)
        value_errorterm = self.errorterm_model.calc_from_distribution(x_inputs=x_values)
        value_combined = value_function + value_errorterm
        return value_combined
    



class IDependency_setter(metaclass=ABCMeta):
    """
    Abstract base class for setting and coordinating the generation of dependencies in nodes.
    Typically uses the kernel_combination_maker, function_maker, and errorterm_maker .

    This interface defines a method for setting dependencies on node objects. Implementations of this interface are responsible 
    for determining how dependencies are assigned to each node, potentially using different strategies or models.
    """
    @abstractmethod
    def set_dependencies(self, node: nodemaker.Nodeobject) :
     """
        Sets dependencies for a given node.

        Args:
            node (nodemaker.Nodeobject): The node object to set dependencies on.
        """


class Dependency_setter_total_random_distributions(IDependency_setter):
    """
    Implementation of IDependency_setter that assigns dependencies as random distributions.

    This class selects a random distribution from a provided collection of initial value distributions to set dependencies for nodes as a IDistribution object instead a Dependencies object.

    Args:
        value_distributions (_dependencymaker_initial_value_distributions.IInitial_value_distribution_collection): 
            A collection of initial value distributions to be used for setting dependencies.
    """

    def __init__(self, value_distributions: _dependencymaker_initial_value_distributions.IInitial_value_distribution_collection):
        self.value_distributions = value_distributions
    
    def set_dependencies(self, node: nodemaker.Nodeobject):
        """
        Sets a random distribution as a dependency for the given node.

        Args:
            node (nodemaker.Nodeobject): The node object to set the dependency on.

        Returns:
            A randomly selected distribution from the collection.
        """
        distribution = self.value_distributions.get_distribution()
        return distribution

class Dependency_setter_default(IDependency_setter):
    """
    Default implementation of IDependency_setter that coordinates the generation of dependencies for nodes.

    This class uses the kernel_combination_maker, function_maker, and errorterm_maker for creating and assigning dependencies. 
    It combines these components to create a "Dependencies" object.

    Args:
        kernel_combination_maker (_kernelcombinator.IKernelcombinator): Used to create kernel combinations for dependencies.
        function_maker (_functionmaker.IFunction_maker): Used to create function models for dependencies.
        errorterm_maker (_errortermmaker.IErrorterm_maker): Used to create error term models for dependencies.
    """

    def __init__(self,kernel_combination_maker: _kernelcombinator.IKernelcombinator, function_maker: _functionmaker.IFunction_maker, errorterm_maker:_errortermmaker.IErrorterm_maker  ):
        self.kernel_combination_maker = kernel_combination_maker
        self.function_maker = function_maker
        self.errorterm_maker = errorterm_maker


    def set_dependencies(self, node: nodemaker.Nodeobject, range_of_output: Tuple[float, float] ):
        """
        Sets dependencies for a node using combined kernel, function, and error term models.

        Args:
            node (nodemaker.Nodeobject): The node object to set dependencies on.
            range_of_output (Tuple[float, float]): The output range for the dependency functions, passed on mainly for normalization settings.

        Returns:
            The generated dependency object for the node.
        """
        if node.source == True: # because sources have no parents
            dimensions = 1
        else : 
            dimensions = len(node.parents)
        kernel = self.kernel_combination_maker.combinate_kernels(dimensions=dimensions)
        function_model = self.function_maker.make_functions(kernel=kernel, errorterm_tolerance =self.errorterm_maker.maximum_tolerance , range_of_output= range_of_output) # function model is a Dependency_functions Object

        errorterm_model = self.errorterm_maker.make_errorterm(function_model=function_model) # should return a dependency_errorterm_object(because in current design Errorterms are capsulated)
        dependency = Dependencies( function_model=function_model, errorterm_model=errorterm_model)
        return dependency







if __name__ == "__main__":
        pass

