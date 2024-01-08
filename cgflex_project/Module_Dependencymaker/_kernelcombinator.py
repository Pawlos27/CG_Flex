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
    """_summary_

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.
    """
    @abstractmethod
    def select_kernels(self, free_dimensions:int, active_dimension_counter):
        """selects appropriate kernels regarding on its dimensioncompability
        Args:
            free_dimensions (int): the number of dimensions left which are free to occupy, relevant for setting the kernel and for selecting the kernel
            active_dimension_counter (_type_): the current dimension which is active at this point in the loop, starting from 0. relevant for setting the kernel parameters

    
        """
    
class Kernel_selector_random(IKernel_selector):
    """Implementation which is selecting kernel randomly from a kernel colelction as long as the requirements fit

    Args:
        kernel_collection (_kernel_collection.IKernel_collection):  setting the kernel colelction from which the selector can choose from
        max_dimensions_per_kernel (int): setting the maximum allowed dimensions per kernel
    """
    def __init__(self, kernel_collection:_kernel_collection.IKernel_collection, max_dimensions_per_kernel:int):
        self.kernel_list = kernel_collection.get_kernel_list()
        self.max_dimensions_per_kernel = max_dimensions_per_kernel
    
    def select_kernels(self, free_dimensions:int, active_dimension_counter):
        """selects a kernel and 
        
        :returns: 
            - kernel - Description of the returned kernel.
            - dimensionality_kernel - Description of the dimensionality_kernel.
        :rtype: (Type of kernel, Type of dimensionality_kernel)
        """
        active_dimension_counter = active_dimension_counter
        limit_used_dimensions = min(free_dimensions,self.max_dimensions_per_kernel) # setting actual limit to the dimensions
        list_kernels_matching = self._make_matching_kernel_list(limit_used_dimensions=limit_used_dimensions) 
        selected_kernel = random.choice(list_kernels_matching)
        print(selected_kernel)
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
        """filters the kernel_list so that only kernelserving the dimension limits will lremain"""
        list_kernels_matching = []
        for kernel in self.kernel_list:
            if kernel.min_dimensions <= limit_used_dimensions:
                list_kernels_matching.append(kernel)
        return list_kernels_matching
      
class Kernel_selector_matchmaking(IKernel_selector):
    """possible implementation for prefering kernels with the exact free capacity, so that possibly only one kernel is used """
    def __init__(self, kernel_collection:_kernel_collection.IKernel_collection, max_dimensions_per_kernel:int ):
        pass

    def select_kernels(self, free_dimension:int ):
        pass


           
class IKernelcombinator(metaclass=ABCMeta):
    """Interface for Implementations that create combined kernels"""
    @abstractmethod
    def combinate_kernels(self, dimensions:int):
     """Interface Method"""
    
class Kernelcombinator_random_picking(IKernelcombinator):
    """randomly picks kernel from a list of kernels and combines them using various operations

    Args:
        kernel_operator_collection (_kernel_collection.IKernel_operator_collection): a collection providing the operations for combination of kernels
        kernel_selector (IKernel_selector): selecting 
    """
    def __init__(self , kernel_operator_collection: _kernel_collection.IKernel_operator_collection = _kernel_collection.Kernel__operator_collection_default(), kernel_selector: IKernel_selector = Kernel_selector_random(max_dimensions_per_kernel= 2,kernel_collection= _kernel_collection.Kernel_collection_general_full())):

        self.kernel_selector = kernel_selector
        self.kernel_operator_collection = kernel_operator_collection
    def combinate_kernels(self, dimensions:int):
        """_summary_

        Args:
            dimensions (int): _description_

        Returns:
            _type_: _description_
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
        free_dimension_counter = dimensions
        active_dimension_counter = 0
        list_of_kernels = []
        while free_dimension_counter > 0 :
            selected_kernel, dimensionality = self.kernel_selector.select_kernels(free_dimensions= free_dimension_counter, active_dimension_counter=active_dimension_counter)
            list_of_kernels.append(selected_kernel)
            active_dimension_counter = active_dimension_counter + dimensionality
            free_dimension_counter = free_dimension_counter - dimensionality
        return list_of_kernels

class Kernelcombinator_linear(IKernelcombinator):
    def __init__(self , kernel_operator_collection: _kernel_collection.IKernel_operator_collection, kernel_selector: IKernel_selector):
    
        self.kernel_selector = kernel_selector
        self.kernel_operator_collection = kernel_operator_collection

    def combinate_kernels(self, dimensions:int):
        pass


