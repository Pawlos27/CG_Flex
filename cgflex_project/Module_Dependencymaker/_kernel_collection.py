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
    @abstractmethod
    def set_kernel(self, input_dim:int, active_dims: list):
     """Interface Method"""


class Rbf_kernel(IKernels):
   def __init__(self, lenghtscale_generator: distributions.IDistributions  ):
        #self.kernel = None
        self.min_dimensions= 1
        self.max_dimensions = float('inf')
        self.lenghtscale_generator = lenghtscale_generator

   def set_kernel(self, input_dim:int, active_dims: list):
        self.kernel = GPy.kern.RBF(input_dim=input_dim, active_dims=active_dims, lengthscale=self.lenghtscale_generator.get_value_from_distribution())





class Periodic_kernel(IKernels ):
   def __init__(self,lenghtscale_generator: distributions.IDistributions  ):
        #self.kernel = None
        self.min_dimensions= 1
        self.max_dimensions = 1
        self.lenghtscale_generator = lenghtscale_generator

   def set_kernel(self, input_dim:int, active_dims: list):
        self.kernel = GPy.kern.PeriodicExponential(input_dim=input_dim, active_dims=active_dims ,lengthscale=self.lenghtscale_generator.get_value_from_distribution())

class Matern32_kernel(IKernels ):
   def __init__(self,lenghtscale_generator: distributions.IDistributions  ):
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
   def __init__(self, lenghtscale_generator: distributions.IDistributions  ):
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
        """ returns the kernel collectiion, represented by the kernel list 

        Returns:
            List[IKernels]: a list
        """

class Kernel_collection_linear(IKernel_collection):
    def __init__(self):
        self.kernellist= []
    def get_kernel_list(self):
       return self.kernellist

class Kernel_collection_rbf_small_lenghthscales(IKernel_collection):
    def __init__(self):
        self.kernellist= [Rbf_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.01,max=0.1))]
    def get_kernel_list(self) :
       return self.kernellist
    
class Kernel_collection_periodic_decrease(IKernel_collection):
    def __init__(self):
        self.kernellist= [Periodic_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.1,max=1))]
    def get_kernel_list(self) :
       return self.kernellist
    
class Kernel_collection_periodic(IKernel_collection):
    def __init__(self):
        self.kernellist= [Periodic_clean_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.1,max=1))]
    def get_kernel_list(self) :
       return self.kernellist
    
class Kernel_collection_matern32(IKernel_collection):
    def __init__(self):
        self.kernellist= [Matern32_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.1,max=1))]
    def get_kernel_list(self) :
       return self.kernellist

class Kernel_collection_variance_default(IKernel_collection):
    def __init__(self ):
        self.kernellist=[Linear_kernel(), Rbf_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.2,max=2)),]
    def get_kernel_list(self):
       return self.kernellist
    
class Kernel_collection_general_default(IKernel_collection):
    def __init__(self):
        self.kernellist=[Linear_kernel(),  Rbf_kernel(lenghtscale_generator=distributions.Distrib_custom_with_valuelist(valuelist=[0.1,0.3,0.5,0.7,1,2])), Matern32_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.1,max=1))]
    def get_kernel_list(self):
       return self.kernellist
    
class Kernel_collection_general_full(IKernel_collection):
    def __init__(self):
        self.kernellist=[Linear_kernel(),  Rbf_kernel(lenghtscale_generator=distributions.Distrib_custom_with_valuelist(valuelist=[0.1,0.2,0.01,0.3,0.05,0.3,0.5,0.7,])), Periodic_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.01,max=1)),Rbf_kernel(lenghtscale_generator=distributions.Distribution_normal_truncated_at_3sigma(mu=0.2, sigma=0.05)),Periodic_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.001,max=1)), Matern32_kernel(lenghtscale_generator= distributions.Distribution_uniform(min=0.001,max=0.6))]
    def get_kernel_list(self):
       return self.kernellist

class Kernel_collection_custom(IKernel_collection):
    def __init__(self, kernellist: List[IKernels]):
        self.kernellist = kernellist
    def get_kernel_list(self):
       return self.kernellist
        


class IKernel_operator(metaclass=ABCMeta):
    """Abstract base class as Interface, its implementations provide oprators and functionalityfor combining kernels"""

    @abstractmethod
    def perform_operation(self, kernel_inputs:list):
        """this method performs the combination operations

        Args:
            kernel_inputs (list): the list contains kernel objects, typicaly there are 2

        Returns:
            kernel: returns a combined kernel
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
    @abstractmethod
    def get_random_operator(self):
     """Interface Method"""


class Kernel__operator_collection_default(IKernel_operator_collection):
    def __init__(self):
        self.operator_list=[Kernel_multiplying(), Kernel_adding()]
    def get_random_operator(self):
       random_operator = random.choice(self.operator_list)
       return random_operator

class Kernel__operator_solo_addition(IKernel_operator_collection):
    def __init__(self):
        self.operator_list= [Kernel_adding()]
    def get_random_operator(self):
       random_operator = self.operator_list[0]
       return random_operator

class Kernel__operator_solo_multiplication(IKernel_operator_collection):
    def __init__(self):
        self.operator_list= [Kernel_multiplying()]
    def get_random_operator(self):
       random_operator = self.operator_list[0]
       return random_operator

class Kernel_operator_collection_shifted_probability(IKernel_operator_collection):
    def __init__(self, probability_for_multiplying):
        self.operator_list=[Kernel_multiplying(), Kernel_adding()]
        self.probability_for_multiplying = probability_for_multiplying
    def get_random_operator(self):
       random_number = np.random.uniform()
       if random_number > self.probability_for_multiplying:
          return self.operator_list[1]
       else:
          return self.operator_list[0]
       
