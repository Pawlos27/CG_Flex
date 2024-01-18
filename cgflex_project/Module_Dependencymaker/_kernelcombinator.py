"""this module contains the INterfaces and Implementations of IKernel_selector and IKernelcombinator, they 
provide functionality to select appopriate kernels and combine them to a combined kernel"""


import numpy as np
import GPy
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import cgflex_project.Module_Dependencymaker._kernel_collection as _kernel_collection
import random
from typing import Any, List, Type

# problem mit enkapsulierung, und kerneleinstellen



class IKernel_selector(metaclass=ABCMeta):
    """
    Interface for selecting a kernel depending on dimension compatibility requirements."""

    @abstractmethod
    def select_kernels(self, free_dimensions:int, active_dimension_counter):
       """
        Selects a kernel based on the current free dimensions and the active dimension counter.

        Args:
            free_dimensions (int): The number of dimensions that are yet to be assigned.
            active_dimension_counter (int): The current count of active dimensions.
        """
        
# class Kernel_selector_matchmaking(IKernel_selector):  """possible implementation for prefering kernels with the exact free capacity, so that possibly only one kernel is used """


class Kernel_selector_random(IKernel_selector):
    """
         Randomly selects a kernel from a specified collection, adhering to dimension coverage requirements.

        Args:
            kernel_collection (IKernel_collection): The collection of available kernels.
            max_dimensions_per_kernel (int): The maximum number of dimensions a kernel can cover.
        """

    def __init__(self, kernel_collection:_kernel_collection.IKernel_collection, max_dimensions_per_kernel:int =1):
        self.kernel_list = kernel_collection.get_kernel_list()
        self.max_dimensions_per_kernel = max_dimensions_per_kernel
    
    def select_kernels(self, free_dimensions:int, active_dimension_counter):
        """
        Selects a random kernel from the available list, ensuring it meets dimension coverage requirements.

        Args:
            free_dimensions (int): The number of dimensions that are yet to be assigned to a kernel.
            active_dimension_counter (int): The current count of active dimensions already assigned.

        Returns:
            GPy.kern.Kern: The selected GPy kernel object, maybe another object if we implement other kernels and GPs.
            int: The number of dimensions assigned to the selected kernel.
        """

        active_dimension_counter = active_dimension_counter
        limit_used_dimensions = min(free_dimensions,self.max_dimensions_per_kernel) # setting actual limit to the dimensions
        list_kernels_matching = self._make_matching_kernel_list(limit_used_dimensions=limit_used_dimensions) 
        selected_kernel = random.choice(list_kernels_matching)
        dimensionality_kernel = random.randint(a=1, b=limit_used_dimensions)
        if selected_kernel.min_dimensions == selected_kernel.max_dimensions:
            dimensionality_kernel = selected_kernel.min_dimensions

        actual_used_dims_for_kernel = []
        for i in range (dimensionality_kernel):
            used_dimension = active_dimension_counter
            actual_used_dims_for_kernel.append(used_dimension)
            active_dimension_counter += 1    

        neuer_kernel= selected_kernel.set_kernel( input_dim=dimensionality_kernel, active_dims=actual_used_dims_for_kernel)
        neuer_kernel= selected_kernel.kernel
        return neuer_kernel, dimensionality_kernel
    
    def _make_matching_kernel_list(self, limit_used_dimensions)->List[_kernel_collection.IKernels]:
        """
        Filters the kernel list to include only those kernels that match the dimension limits.

        Args:
            limit_used_dimensions (int): The maximum number of dimensions that can be assigned to a kernel.

        Returns:
            List[_kernel_collection.IKernels]: A list of kernels that match the given dimension limit.
        """
        list_kernels_matching = []
        for kernel in self.kernel_list:
            if kernel.min_dimensions <= limit_used_dimensions:
                list_kernels_matching.append(kernel)
        return list_kernels_matching
      
           
class IKernelcombinator(metaclass=ABCMeta):
    """
        Generates a combined kernel by selecting kernels for the given number of dimensions.

        Args:
            dimensions (int): The total number of dimensions to be covered by the combined kernel.
        """

    @abstractmethod
    def combinate_kernels(self, dimensions:int):
        """
        Generates a combined kernel for the given number of dimensions.

        Args:
            dimensions (int): The total number of dimensions to be covered by the combined kernel.
        """
    
class Kernelcombinator_random_picking(IKernelcombinator):
    """
    Combines kernels randomly to cover all input parameters, producing a single composite kernel.

    Args:
        kernel_operator_collection (IKernel_operator_collection): The collection of kernel combination operators.
        kernel_selector (IKernel_selector): The selector for choosing individual kernels.
    """

    def __init__(self , kernel_operator_collection: _kernel_collection.IKernel_operator_collection = _kernel_collection.Kernel__operator_collection_default(), kernel_selector: IKernel_selector = Kernel_selector_random(max_dimensions_per_kernel= 2,kernel_collection= _kernel_collection.Kernel_collection_general_full())):

        self.kernel_selector = kernel_selector
        self.kernel_operator_collection = kernel_operator_collection
    def combinate_kernels(self, dimensions:int):
        """
        Combines kernels to cover the specified number of dimensions. 
        It selects kernels randomly and combines them using random operators.

        Args:
            dimensions (int): The total number of dimensions to be covered by the combined kernel.

        Returns:
            GPy.kern.Kern: The final combined GPy kernel object covering all dimensions.
        """
        list_of_kernels = self._make_kernel_list(dimensions= dimensions)

        while len(list_of_kernels) > 1 :
            kernel_1 = random.choice(list_of_kernels)
            list_of_kernels.remove(kernel_1)
            kernel_2 = random.choice(list_of_kernels)
            list_of_kernels.remove(kernel_2)

            output_kernel = self.kernel_operator_collection.get_random_operator().perform_operation( kernel_inputs= [kernel_1, kernel_2])
            list_of_kernels.append(output_kernel)
        
        final_kernel = list_of_kernels[0]
        return final_kernel

    def _make_kernel_list(self, dimensions):
        """
        Creates a list of kernels that collectively cover the specified number of dimensions.

        Args:
            dimensions (int): The number of dimensions to be covered by the kernels.

        Returns:
            List[GPy.kern.Kern]: A list of GPy kernel objects.
        """
        free_dimension_counter = dimensions
        active_dimension_counter = 0
        list_of_kernels = []
        while free_dimension_counter > 0 :
            selected_kernel, dimensionality = self.kernel_selector.select_kernels(free_dimensions= free_dimension_counter, active_dimension_counter=active_dimension_counter)
            list_of_kernels.append(selected_kernel)
            active_dimension_counter = active_dimension_counter + dimensionality
            free_dimension_counter = free_dimension_counter - dimensionality
        return list_of_kernels

