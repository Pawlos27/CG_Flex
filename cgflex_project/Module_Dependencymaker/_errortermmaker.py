from abc import ABCMeta, abstractstaticmethod, abstractmethod
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Type, Union
import cgflex_project.Module_Dependencymaker._errortermmaker_collection as _errortermmaker_collection
import cgflex_project.Module_Dependencymaker._functionmaker as _functionmaker
from dataclasses import dataclass
import copy




class IErrorterm_maker(metaclass=ABCMeta):
    """
    An abstract base class for creating error term models for dependencies in nodes.

    This interface defines the structure and essential methods for creating error terms
    that can be used in dependency models within a graph.
    """
    @abstractmethod
    def make_errorterm(selfself, function_model: _functionmaker.Dependency_functions) -> _errortermmaker_collection.IErrordistribution:
        """Creates and returns an uniquely parameterized error term model 

        Args:
            - function_model (_functionmaker.Dependency_functions): Erroterm model with variable variance needs a functionmodel.

        Returns:
            - _errortermmaker_collection.IErrordistribution: An IErrordistribution object representing the error term distribution.
        """  

class Errorterm_maker_default(IErrorterm_maker):
    """
    Default implementation of the IErrorterm_maker interface. It constructs an error term model 
    for each node separately, ensuring unique parameterization for each node, even if the error term 
    types are the same. This uniqueness is achieved using deepcopy on the distribution object.

    Attributes:
        errorterm_collection_list (List[_errortermmaker_collection.IErrordistribution]): 
            A list of error term distribution models.
        maximum_tolerance (float): 
            The maximum relative tolerance, as a percentage of the total value range of the base function.
    """
    def __init__(self , errorterm_collection: _errortermmaker_collection.IError_term_collection, maximum_tolerance:float = 0.1): # maximum relative tolarce is a percentage of the total value range of the base function
        self.errorterm_collection_list = errorterm_collection.get_errorterm_list()
        self.maximum_tolerance = maximum_tolerance
      
    def make_errorterm(self, function_model: _functionmaker.Dependency_functions) ->  _errortermmaker_collection.IErrordistribution:
        
        
        dimensionality = function_model.functions[0].function_model.return_kernel_dimensions()
        maximum_total_deviation= self._calculate_total_deviation(function_model=function_model)
        original_distribution = random.choice(self.errorterm_collection_list)
        selected_errorterm_distribution = copy.deepcopy(original_distribution)
        
        # when the errorterm is a normal function with variable sigma, we need to pass the range of possible inputs for its function
        if isinstance(selected_errorterm_distribution , _errortermmaker_collection.Error_distribution_normal_variable_variance):
            selected_errorterm_distribution.set_range(range_of_output=function_model.range_of_output)

        selected_errorterm_distribution.make_distribution(dimensionality=dimensionality, maximum_total_deviation=maximum_total_deviation)
        return selected_errorterm_distribution
           

    def _calculate_total_deviation(self,function_model: _functionmaker.Dependency_functions):
        range_total_deviation = function_model.range_of_output[1] - function_model.range_of_output[0]
        total_deviation = range_total_deviation * self.maximum_tolerance
        return total_deviation
    

        




