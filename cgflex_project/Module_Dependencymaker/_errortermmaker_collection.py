"""note, esseciant functions are outsourced in another module"""

from abc import ABCMeta, abstractstaticmethod, abstractmethod
import math
import random
import cgflex_project.Module_Dependencymaker._kernel_collection as _kernel_collection
import numpy as np
import cgflex_project.Module_Dependencymaker._kernelcombinator as _kernelcombinator
import numpy as np
import GPy
import matplotlib.pyplot as plt
from typing import Any, List, Type, Union
import cgflex_project.Module_Dependencymaker._functionmaker as _functionmaker

import cgflex_project.Module_Dependencymaker._errortermmaker_complex_distributions as _errortermmaker_complex_distributions
import cgflex_project.Shared_Classes.distributions as distributions
import cgflex_project.Module_Dependencymaker._functionmaker_extreme_values as _functionmaker_extreme_values
import cgflex_project.Module_Dependencymaker._inputloader as _inputloader
from copy import deepcopy





class IErrordistribution(metaclass=ABCMeta): # all distributions are designed so they fall into >0, 

    @abstractmethod
    def make_distribution(self,dimensionality:int, maximum_total_deviation:float ):
     """Interface Method"""
    @abstractmethod
    def calc_from_distribution(self, x_inputs:list):
        pass
    @abstractmethod
    def show_error_distribution(self, resolution, label):
        pass


class Error_distribution_no_error(IErrordistribution):

    def __init__(self):
        pass

    def make_distribution(self,dimensionality:int, maximum_total_deviation:float ):
        pass

    def calc_from_distribution(self, x_inputs):
        error_value = 0
        return error_value
    def show_error_distribution(self, label="errorterm"):
        print(f" No Errorterm")


class Error_distribution_normal(IErrordistribution):

    def __init__(self):
        pass

    def make_distribution(self,dimensionality:int, maximum_total_deviation:float ):
        self.maximum_total_deviation= maximum_total_deviation
        maximum_sigma = maximum_total_deviation/6
        sigma = np.random.uniform(low=0, high=maximum_sigma)
        mu = sigma*3
        self.distribution = distributions.Distribution_normal(mu=mu, sigma=sigma)
        #print(f"maximum_sigma is {maximum_sigma} and actual sigma is {sigma}")

    def calc_from_distribution(self, x_inputs):
        error_value = self.distribution.get_value_from_distribution()
        if error_value > self.maximum_total_deviation or error_value< 0:
            error_value = self.distribution.get_value_from_distribution()
        return error_value
    def show_error_distribution(self, label="errorterm"):
        self.distribution.plot_distribution(label=label + f" Distribution: {self.__class__.__name__}")


class Error_distribution_normal_variable_variance(IErrordistribution):

    def __init__(self, function_maker: _functionmaker.IFunction_maker, kernel_maker: _kernelcombinator.IKernelcombinator, ):
        self.function_maker = function_maker
        self.kernel_maker = kernel_maker
        
    def make_distribution(self,dimensionality:int,  maximum_total_deviation:float  ):
        self.maxium_total_deviation = maximum_total_deviation
        kernel = self.kernel_maker.combinate_kernels(dimensions=dimensionality)
        maximum_sigma = maximum_total_deviation/6
        self.function_model= self.function_maker.make_functions(kernel=kernel,errorterm_tolerance=0, range_of_output=(0,maximum_sigma))
    def calc_from_distribution(self, x_inputs:list):
        sigma_normalized = self.function_model.calculate_value(x_inputs=x_inputs)
        if sigma_normalized < 0:
            sigma_normalized = 0
            print("attention,sigma in errortermn with variable variance was <0")
        mu = sigma_normalized*3
        error_value = self.distribution = distributions.Distribution_normal_truncated_at_3sigma(mu=mu, sigma=sigma_normalized).get_value_from_distribution()
        return error_value
    def show_error_distribution(self, label="errorterm" ):
        self.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=30, label= label + f" Distribution: {self.__class__.__name__}" )


class Error_distribution_mixture_model(IErrordistribution):

    def __init__(self, mixture_list_of_mus_maker = _errortermmaker_complex_distributions.Complex_distribution_list_of_Mus_maker):
        self.mixture_list_of_mus_maker = mixture_list_of_mus_maker
    def make_distribution(self,dimensionality:int,  maximum_total_deviation:float ):
        self.maximum_total_deviation = maximum_total_deviation
        lists_of_mus =self.mixture_list_of_mus_maker.sample_lists_of_mus(size_of_samples=1,maximum_total_deviation=maximum_total_deviation)
        sigma = self.mixture_list_of_mus_maker.return_sigma()
        # sigma is maximum_distance/5
        self.distribution = distributions.Distribution_mixture_of_normals_truncated_at_3sigma(list_of_mus=lists_of_mus[0], sigma=sigma)

    def calc_from_distribution(self, x_inputs):
        error_value = self.distribution.get_value_from_distribution()
        if error_value > self.maximum_total_deviation:
            error_value = self.maximum_total_deviation
        return error_value
    def show_error_distribution(self, label="errorterm" ):
        self.distribution.plot_distribution( label= label + f" Distribution: {self.__class__.__name__}")


class Error_distribution__mixture_model_interpolated(IErrordistribution):

    def __init__(self,interpolator:_errortermmaker_complex_distributions.IInterpolation_model, mixture_list_of_mus_maker : _errortermmaker_complex_distributions.Complex_distribution_list_of_Mus_maker ):
        self.mixture_list_of_mus_maker = mixture_list_of_mus_maker
        self.interpolator = interpolator()
    def make_distribution(self,dimensionality:int,  maximum_total_deviation:float  ):
        """central to the mixture model is the list of mus , it represents the position of its components, in our interpolated version, each component position is represented by an interpolation model dependent on the 
        input_values, in the first step we need to train a couple of mixtures ech represented by a  list_of_mus.
          these cann then further be used to train the model  """
        list_of_mu_as_interpolation_models = []
        size_of_samples_mus = self.interpolator.return_number_of_required_values(dimensions=dimensionality)
        self.maximum_total_deviation = maximum_total_deviation
        lists_of_mus_training = self.mixture_list_of_mus_maker.sample_lists_of_mus(size_of_samples=size_of_samples_mus,maximum_total_deviation=maximum_total_deviation)
        self.lists_of_mus_training = lists_of_mus_training
        number_of_components = len(lists_of_mus_training[0])
        print(number_of_components)
        #we are training a model for each component of the interpolated mixture, the training points are from the representations of the component in the different sample mixtures
        for i in range(number_of_components):
            training_values_per_component = []
            model = deepcopy(self.interpolator)
            for mixture in lists_of_mus_training:
                value = mixture[i]
                training_values_per_component.append(value)
            model.set_interpolator_model(values= training_values_per_component, dimensions= dimensionality)
            list_of_mu_as_interpolation_models.append(model)
        self.list_of_mu_as_interpolation_models = list_of_mu_as_interpolation_models
        self.sigma = self.mixture_list_of_mus_maker.return_sigma()

    def calc_from_distribution(self, x_inputs):
        list_of_mus_interpolated = []
        for interpolation_model in self.list_of_mu_as_interpolation_models:
            value_mu = interpolation_model.calculate_interpolated_values( inputs= np.array([x_inputs]))
            list_of_mus_interpolated.append(value_mu[0])
        distribution = distributions.Distribution_mixture_of_normals_truncated_at_3sigma(list_of_mus=list_of_mus_interpolated, sigma=self.sigma)
        error_value = distribution.get_value_from_distribution()
        distribution.plot_distribution()
        if error_value > self.maximum_total_deviation:
            error_value = self.maximum_total_deviation
        return error_value
    def show_error_distribution(self, label="errorterm"):

        self.list_of_mu_as_interpolation_models[49].plot_interpolator_model()
        distribution = distributions.Distribution_mixture_of_normals_truncated_at_3sigma(list_of_mus=self.lists_of_mus_training[0], sigma=self.sigma)
        distribution.plot_distribution( label= label + f" Distribution: {self.__class__.__name__}")

        """currently only showing one mixture, and interpolation mesh vor the possible positions of the first element of the mixture regarding to its training points"""
    def show_mixture_interpolated_for_certain_point(self, x_inputs):
        list_of_mus_interpolated = []
        for interpolation_model in self.list_of_mu_as_interpolation_models:
            value_mu = interpolation_model.calculate_interpolated_values( inputs= np.array([x_inputs]))
            list_of_mus_interpolated.append(value_mu[0])
        distribution = distributions.Distribution_mixture_of_normals_truncated_at_3sigma(list_of_mus=list_of_mus_interpolated, sigma=self.maximum_distance/1.5)
        distribution.plot_distribution()




class IError_term_collection(metaclass=ABCMeta):
    @abstractmethod
    def get_errorterm_list(self) -> List[IErrordistribution]:
     """Interface Method"""

class Error_term_collection_solo_no_errorterm(IError_term_collection):

    def __init__(self):
        self.error_term_list = [ Error_distribution_no_error() ]   
    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list                 

class Error_term_collection_solo_normal_variable_variance(IError_term_collection):

    def __init__(self):
        self.error_term_list = [ Error_distribution_normal_variable_variance(function_maker= _functionmaker.Function_maker_evenly_discontinuity_in_one_dimension(error_term_modus= True,inputloader= _inputloader. Inputloader_for_solo_random_values(), normalizer=_functionmaker_extreme_values.Normalizer_minmax_stretch() ,discontinuity_frequency= 0.7,  maximum_discontinuities=2, discontinuity_reappearance_frequency=0.5, extreme_value_setter=_functionmaker_extreme_values.Extreme_value_setter_solo_dimensionmax(resolution=100)), 
                                                                                                                                                                   kernel_maker=_kernelcombinator.Kernelcombinator_random_picking( kernel_operator_collection=_kernel_collection.Kernel__operator_collection_default(), kernel_selector= _kernelcombinator.Kernel_selector_random(max_dimensions_per_kernel= 2,kernel_collection= _kernel_collection.Kernel_collection_general_default())))]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_solo_mixture_model(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_mixture_model(mixture_list_of_mus_maker=_errortermmaker_complex_distributions.Complex_distribution_list_of_Mus_maker(number_of_maximum_modes=5,total_number_of_elements=50,distribution_for_distances_between_components=_errortermmaker_complex_distributions.Errorterm_mixture_distance_distribution_random())) ]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_solo_mixture_model_interpolated(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution__mixture_model_interpolated(mixture_list_of_mus_maker=_errortermmaker_complex_distributions.Complex_distribution_list_of_Mus_maker(number_of_maximum_modes=5,total_number_of_elements=50,distribution_for_distances_between_components=_errortermmaker_complex_distributions.Errorterm_mixture_distance_distribution_random()),interpolator=_errortermmaker_complex_distributions.Interpolation_model_RBFInterpolator_datapoints_random) ]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list
    
class Error_term_collection_solo_normal_multimodal(IError_term_collection):

    def __init__(self):
        self.error_term_list = [ Error_distribution_mixture_model(number_of_elements_max=1, number_of_maximum_modes=5,distribution_for_distances_between_components=_errortermmaker_complex_distributions.Errorterm_mixture_distance_distribution_random())]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list



class Error_term_collection_solo_normal(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_normal()]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        return self.error_term_list
    
class Error_term_collection_Tier_2_default(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_normal(), Error_distribution_normal_variable_variance(function_maker= _functionmaker.Function_maker_evenly_discontinuity_in_one_dimension(discontinuity_frequency= 0.7,  maximum_discontinuities=2, discontinuity_reappearance_frequency=0.5, extreme_value_setter=_functionmaker_extreme_values.Extreme_value_setter_solo_dimensionmax(resolution=100)),
                                                                                                                                                                   kernel_maker=_kernelcombinator.Kernelcombinator_random_picking( kernel_operator_collection=_kernel_collection.Kernel__operator_collection_default(), kernel_selector= _kernelcombinator.Kernel_selector_random(max_dimensions_per_kernel= 2,kernel_collection= _kernel_collection.Kernel_collection_general_default())))]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_Tier_5_default(IError_term_collection):

    def __init__(self):
        self.error_term_list=[Error_distribution_normal(),Error_distribution_mixture_model(mixture_list_of_mus_maker=_errortermmaker_complex_distributions.Complex_distribution_list_of_Mus_maker(number_of_maximum_modes=5,total_number_of_elements=50,distribution_for_distances_between_components=_errortermmaker_complex_distributions.Errorterm_mixture_distance_distribution_random())) ,
                              Error_distribution_normal_variable_variance(function_maker= _functionmaker.Function_maker_evenly_discontinuity_in_one_dimension(error_term_modus= True, inputloader= _inputloader. Inputloader_for_solo_random_values(),normalizer= _functionmaker_extreme_values.Normalizer_minmax_stretch(),discontinuity_frequency= 0.7,  maximum_discontinuities=2, discontinuity_reappearance_frequency=0.5, extreme_value_setter=_functionmaker_extreme_values.Extreme_value_setter_solo_dimensionmax(resolution=300)),
                                                                                                                                                                   kernel_maker=_kernelcombinator.Kernelcombinator_random_picking( kernel_operator_collection=_kernel_collection.Kernel__operator_collection_default(), kernel_selector= _kernelcombinator.Kernel_selector_random(max_dimensions_per_kernel= 2,kernel_collection= _kernel_collection.Kernel_collection_general_default())))]
                                                                                                                                                                  
    def get_errorterm_list(self) -> List[IErrordistribution]:
        return self.error_term_list

class Error_term_collection_custom(IError_term_collection):

    def __init__(self,error_term_list:List[IError_term_collection] ):
        self.error_term_list=error_term_list

    def get_errorterm_list(self) -> List[IErrordistribution]:
        return self.error_term_list











if __name__ == "__main__":

    pass