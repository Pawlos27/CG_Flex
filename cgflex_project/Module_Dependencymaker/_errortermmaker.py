from abc import ABCMeta, abstractstaticmethod, abstractmethod

import random


import numpy as np

import matplotlib.pyplot as plt
from typing import Any, List, Type, Union
import cgflex_project.Module_Dependencymaker._errortermmaker_collection as _errortermmaker_collection
import cgflex_project.Module_Dependencymaker._functionmaker as _functionmaker
from dataclasses import dataclass
import copy



@dataclass
class Dependency_errorterm:
    errorterm_distribution: _errortermmaker_collection.IErrordistribution
    errorterm_maximum: float
     


class IErrorterm_maker(metaclass=ABCMeta):
    @abstractmethod
    def make_errorterm(selfself, function_model: _functionmaker.Dependency_functions) -> _errortermmaker_collection.IErrordistribution:
     """Interface Method"""    

class Errorterm_maker_default(IErrorterm_maker):
    def __init__(self , errorterm_collection: _errortermmaker_collection.IError_term_collection, maximum_tolerance:float = 0.1): # maximum relative tolarce is a percentage of the total value range of the base function
        self.errorterm_collection_list = errorterm_collection.get_errorterm_list()
        self.maximum_tolerance = maximum_tolerance
      
    def make_errorterm(self, function_model: _functionmaker.Dependency_functions) -> Dependency_errorterm:
        
        
        dimensionality = function_model.functions[0].function_model.return_kernel_dimensions()
        maximum_total_deviation= self._calculate_total_deviation(function_model=function_model)
        original_distribution = random.choice(self.errorterm_collection_list)
        selected_errorterm_distribution = copy.deepcopy(original_distribution)

        selected_errorterm_distribution.make_distribution(dimensionality=dimensionality, maximum_total_deviation=maximum_total_deviation)
        dependency_errorterm_object = Dependency_errorterm(errorterm_distribution=selected_errorterm_distribution , errorterm_maximum=maximum_total_deviation)
        return dependency_errorterm_object
           

    def _calculate_total_deviation(self,function_model: _functionmaker.Dependency_functions):
        range_total_deviation = function_model.range_of_output[1] - function_model.range_of_output[0]
        total_deviation = range_total_deviation * self.maximum_tolerance
        return total_deviation

        




