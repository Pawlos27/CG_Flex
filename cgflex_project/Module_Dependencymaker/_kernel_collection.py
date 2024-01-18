import numpy as np
import GPy
import matplotlib.pyplot as plt
import random
from typing import Any, List, Type
import cgflex_project.Shared_Classes.distributions as distributions
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import numpy as np
import scipy.stats as stats



class IKernels(metaclass=ABCMeta):
    """
    Abstract base class for ecapsualting different kernel types.
    """
    @abstractmethod
    def set_kernel(self, input_dim:int, active_dims: list):
     """
        Sets the kernel with the  specified number "input dimensions" and the related explicit "active dimensions".

        Args:
            input_dim (int): Number of input dimensions.
            active_dims (list): List of active dimensions for the kernel.
        """


class Rbf_kernel(IKernels):
   """
    Represents a Radial Basis Function (RBF) kernel.

    Args:
        lenghtscale_generator (distributions.IDistributions): Distribution to generate lengthscale.
    """
   def __init__(self, lenghtscale_generator: distributions.IDistributions = distributions.Distribution_uniform(min=0.01,max=1)  ):
        #self.kernel = None
        self.min_dimensions= 1
        self.max_dimensions = float('inf')
        self.lenghtscale_generator = lenghtscale_generator

   def set_kernel(self, input_dim:int, active_dims: list):
        self.kernel = GPy.kern.RBF(input_dim=input_dim, active_dims=active_dims, lengthscale=self.lenghtscale_generator.get_value_from_distribution())





class Periodic_kernel_decreasing(IKernels ):
   def __init__(self,lenghtscale_generator: distributions.IDistributions = distributions.Distribution_uniform(min=0.01,max=1) ):
        #self.kernel = None
        self.min_dimensions= 1
        self.max_dimensions = 1
        self.lenghtscale_generator = lenghtscale_generator

   def set_kernel(self, input_dim:int, active_dims: list):
        self.kernel = GPy.kern.PeriodicExponential(input_dim=input_dim, active_dims=active_dims ,lengthscale=self.lenghtscale_generator.get_value_from_distribution())

class Matern32_kernel(IKernels ):
   def __init__(self,lenghtscale_generator: distributions.IDistributions = distributions.Distribution_uniform(min=0.01,max=1)  ):
        #self.kernel = None
        self.min_dimensions= 1
        self.max_dimensions = 1
        self.lenghtscale_generator = lenghtscale_generator

   def set_kernel(self, input_dim:int, active_dims: list):
        self.kernel = GPy.kern.Matern32(input_dim=input_dim, active_dims=active_dims ,lengthscale=self.lenghtscale_generator.get_value_from_distribution())


class Linear_kernel(IKernels):
   def __init__(self):
        #self.kernel = None
        self.min_dimensions= 1
        self.max_dimensions = float('inf')

   def set_kernel(self, input_dim:int, active_dims: list):
        self.kernel = GPy.kern.Linear(input_dim=input_dim, active_dims=active_dims )


class Periodic_clean_kernel(IKernels):
   def __init__(self, lenghtscale_generator: distributions.IDistributions = distributions.Distribution_uniform(min=0.01,max=1)  ):
        #self.kernel = None
        self.min_dimensions= 1
        self.max_dimensions = float('inf')
        self.lenghtscale_generator = lenghtscale_generator

   def set_kernel(self, input_dim:int, active_dims: list):
        self.kernel = GPy.kern.Linear(input_dim=input_dim, active_dims=active_dims )
        self.kernel = GPy.kern.StdPeriodic(input_dim=input_dim, active_dims=active_dims , lengthscale=self.lenghtscale_generator.get_value_from_distribution(), period=self.lenghtscale_generator.get_value_from_distribution())





#kernel collections stores dictionary/list of kernel objects and returns random ones
class IKernel_collection(metaclass=ABCMeta):
    """Abstract Class as an Interface, its implementations provide collections of kernels
    """
    @abstractmethod
    def get_kernel_list(self)->List[IKernels]:
        """
        Returns a list of kernel objects in the collection.

        Returns:
            List[IKernels]: List of kernel objects.
        """


class Kernel_collection_linear(IKernel_collection):
    """containing only the linear kernel"""
    def __init__(self):
        self.kernellist= []
    def get_kernel_list(self):
       return self.kernellist

class Kernel_collection_rbf_small_lenghthscales(IKernel_collection):
    """containing only the rbf kernel with small lenghtscales"""
    def __init__(self):
        self.kernellist= [Rbf_kernel()]
    def get_kernel_list(self) :
       return self.kernellist
    
class Kernel_collection_periodic_decreasing(IKernel_collection):
    """containing only the preiodic decreasing kernel """
    def __init__(self):
        self.kernellist= [Periodic_kernel_decreasing()]
    def get_kernel_list(self) :
       return self.kernellist
    
class Kernel_collection_periodic(IKernel_collection):
    """containing only the preiodic kernel """
    def __init__(self):
        self.kernellist= [Periodic_clean_kernel()]
    def get_kernel_list(self) :
       return self.kernellist
    
class Kernel_collection_matern32(IKernel_collection):
    """containing only the matern32 kernel """
    def __init__(self):
        self.kernellist= [Matern32_kernel()]
    def get_kernel_list(self) :
       return self.kernellist

class Kernel_collection_variance_default(IKernel_collection):
    """containing only the rbf and linear  kernel """
    def __init__(self ):
        self.kernellist=[Linear_kernel(), Rbf_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.05,max=0.8)),]
    def get_kernel_list(self):
       return self.kernellist
    
class Kernel_collection_general_default(IKernel_collection):
    """containing a mix of kernels """
    def __init__(self):
        self.kernellist=[Linear_kernel(),  Rbf_kernel(lenghtscale_generator=distributions.Distrib_custom_with_valuelist(valuelist=[0.1,0.3,0.01,0.3,0.05,0.5,0.7,1])), Matern32_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.1,max=1))]
    def get_kernel_list(self):
       return self.kernellist
    
class Kernel_collection_general_full(IKernel_collection):
    """containing a mix of kernels """
    def __init__(self):
        self.kernellist=[Linear_kernel(),  Rbf_kernel(lenghtscale_generator=distributions.Distrib_custom_with_valuelist(valuelist=[0.1,0.2,0.01,0.3,0.05,0.3,0.5,0.7,])), Periodic_kernel_decreasing(),Rbf_kernel(),Periodic_clean_kernel(), Matern32_kernel()]
    def get_kernel_list(self):
       return self.kernellist

class Kernel_collection_custom(IKernel_collection):
    """
    Represents a custom collection of kernel objects.

    Args:
        kernellist (List[IKernels]): List of kernel objects to be included in the collection.
    """
    def __init__(self, kernellist: List[IKernels]):
        self.kernellist = kernellist
    def get_kernel_list(self):
       return self.kernellist
        


class IKernel_operator(metaclass=ABCMeta):
    """
    Abstract base class for kernel operators. 
    This interface defines methods for combining Gaussian process kernels.
    """

    @abstractmethod
    def perform_operation(self, kernel_inputs:list):
        """
        Performs a combination operation on a list of kernel objects.

        Args:
            kernel_inputs (list): A list containing kernel objects, typically two kernels.

        Returns:
            kernel: The combined kernel after performing the specified operation.
        """
class Kernel_adding(IKernel_operator):
    """class for summation of kernels """
    def __init__(self):
        pass
    def perform_operation(self, kernel_inputs:list):
        kernel_sum = kernel_inputs[0]  # Start with the first kernel
        for kernel in kernel_inputs[1:]:
            kernel_sum = kernel_sum + kernel 
        return kernel_sum
    
class Kernel_multiplying(IKernel_operator):
    """class for multiplication of kernels from a passed list with each other in sequence"""
    def __init__(self):
        pass
    def perform_operation(self, kernel_inputs:list):
        kernel_product = kernel_inputs[0]  # Start with the first kernel
        for kernel in kernel_inputs[1:]:
            kernel_product = kernel_product * kernel  # Multiply with each subsequent kernel
        return kernel_product






class IKernel_operator_collection(metaclass=ABCMeta):
    """
    Abstract base class for collections of kernel operators.
    """
    @abstractmethod
    def get_random_operator(self) -> IKernel_operator:
     """
        Abstract method to get a random kernel operator.

        Returns:
            Kernel operator object.
        """


class Kernel__operator_collection_default(IKernel_operator_collection):
    """
    Default collection of kernel operators.
    """
    def __init__(self):
        self.operator_list=[Kernel_multiplying(), Kernel_adding()]
    def get_random_operator(self):
       random_operator = random.choice(self.operator_list)
       return random_operator

class Kernel__operator_solo_addition(IKernel_operator_collection):
    """
    Collection of kernel operators containing only the addition operator.
    """
    def __init__(self):
        self.operator_list= [Kernel_adding()]
    def get_random_operator(self):
       random_operator = self.operator_list[0]
       return random_operator

class Kernel__operator_solo_multiplication(IKernel_operator_collection):
    """
    Collection of kernel operators containing only the multiplication operator.
    """
    def __init__(self):
        self.operator_list= [Kernel_multiplying()]
    def get_random_operator(self):
       random_operator = self.operator_list[0]
       return random_operator

class Kernel_operator_collection_shifted_probability(IKernel_operator_collection):
    """
    Collection of kernel operators with a configurable probability between multiplication and addition.

    Args:
        probability_for_multiplying (float): Probability for selecting the multiplication operator.
    """
    def __init__(self, probability_for_multiplying:float = 0.3):
        self.operator_list=[Kernel_multiplying(), Kernel_adding()]
        self.probability_for_multiplying = probability_for_multiplying
    def get_random_operator(self):
       random_number = np.random.uniform()
       if random_number > self.probability_for_multiplying:
          return self.operator_list[1]
       else:
          return self.operator_list[0]
       
