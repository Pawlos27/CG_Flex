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
from typing import Any, List, Type, Union, Tuple
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
        self.distribution = distributions.Distribution_normal_truncated_at_3sigma_bound_to_zero_random_at_construction(sigma_max=maximum_total_deviation/6)

    def calc_from_distribution(self, x_inputs):
        error_value = self.distribution.get_value_from_distribution()
        if error_value > self.maximum_total_deviation or error_value< 0:
            error_value = self.distribution.get_value_from_distribution()
        return error_value
    def show_error_distribution(self, label="errorterm"):
        self.distribution.plot_distribution(label=label + f" Distribution: {self.__class__.__name__}")


class Error_distribution_normal_variable_variance(IErrordistribution):

    def __init__(self, function_maker: _functionmaker.IFunction_maker =  _functionmaker.Function_maker_evenly_discontinuity_in_one_dimension() , kernel_maker: _kernelcombinator.IKernelcombinator = _kernelcombinator.Kernelcombinator_random_picking(), normalizer: _functionmaker_extreme_values.INormalizer = _functionmaker_extreme_values.Normalizer_minmax_stretch()):
        self.function_maker = function_maker
        self.kernel_maker = kernel_maker
        self.normalizer = normalizer
    
    def set_range(self, range_of_output= Tuple[int,int]):
        self.range_of_output = range_of_output
        
    def make_distribution(self,dimensionality:int,  maximum_total_deviation:float  ):
        self.maxium_total_deviation = maximum_total_deviation
        kernel = self.kernel_maker.combinate_kernels(dimensions=dimensionality)
        self.function_model= self.function_maker.make_functions(kernel=kernel,errorterm_tolerance=0, range_of_output=self.range_of_output)
        maximum_sigma = maximum_total_deviation/6
        self.normalizer.set_normalizer(output_maximum= maximum_sigma,output_minimum= 0,input_max=self.range_of_output[1],input_min=self.range_of_output[0])
    def calc_from_distribution(self, x_inputs:list):
        sigma_normalized = self.function_model.calculate_value(x_inputs=x_inputs)
        if sigma_normalized < 0:
            sigma_normalized = 0
        error_value = distributions.Distribution_normal_truncated_at_3sigma_bound_to_zero(sigma= sigma_normalized).get_value_from_distribution()
        error_value_normalized = self.normalizer.normalize_value(input_value=error_value)
        return error_value_normalized
    def show_error_distribution(self, label="errorterm" ):
        self.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=30, label= label + f" {self.__class__.__name__}, Function for sigma will be stretched to 0-{self.maxium_total_deviation}" )


class Error_distribution_mixture_model_normal_multimodal(IErrordistribution):

    def __init__(self,number_of_modes: int = 5, fixed_number_of_modes:bool = False)  :
        if number_of_modes <=0:
            raise ValueError("number of modes must be >0")
   
 
        self.mixture_list_of_mus_maker  = distributions.Complex_distribution_list_of_Mus_maker(fixed_number_of_modes= fixed_number_of_modes, number_of_maximum_modes=number_of_modes, multimodal_normal=True)


    def make_distribution(self,dimensionality:int,  maximum_total_deviation:float ):

        self.distribution = distributions.Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(mixture_list_of_mus_maker=self.mixture_list_of_mus_maker, upper_border=maximum_total_deviation)

    def calc_from_distribution(self, x_inputs):
        error_value = self.distribution.get_value_from_distribution()
        return error_value
    def show_error_distribution(self, label="errorterm" ):
        self.distribution.plot_distribution( label= label + f" Distribution: {self.__class__.__name__}")


class Error_distribution_mixture_model_complex(IErrordistribution):

    def __init__(self, mixture_list_of_mus_maker : distributions.Complex_distribution_list_of_Mus_maker = distributions.Complex_distribution_list_of_Mus_maker())  :
        self.mixture_list_of_mus_maker = mixture_list_of_mus_maker
    def make_distribution(self,dimensionality:int,  maximum_total_deviation:float ):

        self.distribution = distributions.Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(mixture_list_of_mus_maker=self.mixture_list_of_mus_maker, upper_border=maximum_total_deviation)


    def calc_from_distribution(self, x_inputs):
        error_value = self.distribution.get_value_from_distribution()
        return error_value
    def show_error_distribution(self, label="errorterm" ):
        self.distribution.plot_distribution( label= label + f" Distribution: {self.__class__.__name__}")


class Error_distribution__mixture_model_interpolated(IErrordistribution):

    def __init__(self,interpolator:_errortermmaker_complex_distributions.IInterpolation_model = _errortermmaker_complex_distributions.Interpolation_model_RBFInterpolator_datapoints_grid , mixture_list_of_mus_maker : _errortermmaker_complex_distributions.Complex_distribution_list_of_Mus_maker = distributions.Complex_distribution_list_of_Mus_maker() ):
        self.mixture_list_of_mus_maker = mixture_list_of_mus_maker
        self.interpolator = interpolator()
    def make_distribution(self,dimensionality:int,  maximum_total_deviation:float  ):
        """central to the mixture model is the list of mus , it represents the position of its components, in our interpolated version, each component position is represented by an interpolation model dependent on the 
        input_values, in the first step we need to train a couple of mixtures ech represented by a  list_of_mus.
          these cann then further be used to train the model  """
        
        self.maximum_total_deviation = maximum_total_deviation
        
        list_of_mu_as_interpolation_models = []
        size_of_samples_mus = self.interpolator.return_number_of_required_values(dimensions=dimensionality)
        self.maximum_total_deviation = maximum_total_deviation
        lists_of_mus_training = self.mixture_list_of_mus_maker.sample_lists_of_mus(size_of_samples=size_of_samples_mus,maximum_total_deviation=maximum_total_deviation)
        self.lists_of_mus_training = lists_of_mus_training
        number_of_components = len(lists_of_mus_training[0])

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
        distribution = distributions.Distribution_mixture_of_normals_truncated_at_3sigma_and_outlier_correction_for_interpolation(list_of_mus=list_of_mus_interpolated, sigma=self.sigma, upper_limit=self.maximum_total_deviation)
        error_value = distribution.get_value_from_distribution()

        if error_value > self.maximum_total_deviation:
            error_value = self.maximum_total_deviation
        return error_value
    def show_error_distribution(self, label="errorterm"):

        self.list_of_mu_as_interpolation_models[0].plot_interpolator_model()
        distribution = distributions.Distribution_mixture_of_normals_truncated_at_3sigma_and_outlier_correction_for_interpolation(list_of_mus=self.lists_of_mus_training[0], sigma=self.sigma, upper_limit=self.maximum_total_deviation)
        distribution.plot_distribution( label= label + f" Distribution: {self.__class__.__name__}")

        """currently only showing one mixture, and interpolation mesh vor the possible positions of the first element of the mixture regarding to its training points"""
    def show_mixture_interpolated_for_certain_point(self, x_inputs):
        list_of_mus_interpolated = []
        for interpolation_model in self.list_of_mu_as_interpolation_models:
            value_mu = interpolation_model.calculate_interpolated_values( inputs= np.array([x_inputs]))
            list_of_mus_interpolated.append(value_mu[0])
        distribution = distributions.Distribution_mixture_of_normals_truncated_at_3sigma_and_outlier_correction_for_interpolation(list_of_mus=list_of_mus_interpolated, sigma=self.sigma, upper_limit=self.maximum_total_deviation)
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

class Error_term_collection_solo_normal(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_normal()]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        return self.error_term_list

class Error_term_collection_solo_normal_multimodal(IError_term_collection):

    def __init__(self):
        self.error_term_list = [ Error_distribution_mixture_model_normal_multimodal() ]
    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_solo_normal_variable_variance(IError_term_collection):

    def __init__(self):
        self.error_term_list = [ Error_distribution_normal_variable_variance()]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_solo_mixture_model(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_mixture_model_complex()]
    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_solo_mixture_model_interpolated(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution__mixture_model_interpolated() ]

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list
    



class Error_term_collection_Tier_1(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_no_error()]                                                                                                                            
    
    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list
    
class Error_term_collection_Tier_2(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_no_error(), Error_distribution_normal()]                                                                                                                                

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_Tier_3(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_no_error(), Error_distribution_normal(),Error_distribution_normal_variable_variance()]                                                                                                                               

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_Tier_4(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_no_error(), Error_distribution_normal(),Error_distribution_normal_variable_variance(), Error_distribution_mixture_model_normal_multimodal()]                                                                                                                                

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_Tier_5(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_no_error(), Error_distribution_normal(),Error_distribution_normal_variable_variance(), Error_distribution_mixture_model_normal_multimodal()]                                                                                                                                 

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_Tier_6(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_no_error(), Error_distribution_normal(),Error_distribution_normal_variable_variance(), Error_distribution_mixture_model_normal_multimodal(), Error_distribution_mixture_model_complex()]                                                                                                                                 

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_Tier_7(IError_term_collection):

    def __init__(self):
        self.error_term_list = [Error_distribution_no_error(), Error_distribution_normal(),Error_distribution_normal_variable_variance(), Error_distribution_mixture_model_normal_multimodal(), Error_distribution_mixture_model_complex(), Error_distribution__mixture_model_interpolated() ]                                                                                                                                 

    def get_errorterm_list(self) -> List[IErrordistribution]:
        list = self.error_term_list
        return list

class Error_term_collection_custom(IError_term_collection):

    def __init__(self,error_term_list:List[IError_term_collection] ):
        self.error_term_list=error_term_list

    def get_errorterm_list(self) -> List[IErrordistribution]:
        return self.error_term_list










if __name__ == "__main__":

    #a = Error_distribution_no_error()
    #a = Error_distribution_normal()
    ###a = Error_distribution_normal_variable_variance()
    #a = Error_distribution_mixture_model_complex()
    #a = Error_distribution_mixture_model_normal_multimodal(number_of_modes=5, fixed_number_of_modes= True)
    a = Error_distribution__mixture_model_interpolated()
    a.make_distribution(dimensionality=1,maximum_total_deviation=0.1)
    a.show_error_distribution()
    value = a.calc_from_distribution(x_inputs=[0.1])
    value = a.calc_from_distribution(x_inputs=[0.2])
    value = a.calc_from_distribution(x_inputs=[0.9])
    print(value)
    pass