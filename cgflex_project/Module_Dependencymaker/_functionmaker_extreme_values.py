
import numpy as np
from typing import Any, List, Type, Optional
import random
from abc import ABCMeta, abstractstaticmethod, abstractmethod
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import GPy

import matplotlib.pyplot as plt
import cgflex_project.Module_Dependencymaker._inputloader as _inputloader
from  cgflex_project.Module_Dependencymaker._functionmaker_gaussian_process_models import IGaussianProcessModel





@dataclass
class Extreme_value_with_coordinates:
    value: float
    coordinates : np.ndarray
    
@dataclass
class Predictions: # predictions may be filled with only 1d-inputs despite being a multidimensional function, in case we have sparse inputs
    dimension:int
    predicted_values: np.ndarray
    input_values : np.ndarray
    function_nr: Optional[int] = None
    normalized: bool = False
    

class IExtreme_value_setter(metaclass=ABCMeta): # interface for strategies for setting extreme values currently only solo_dimension_max.   extreme_value_empty might be usable if extreme value setter is not needed due to already "normalized kernels"
    @abstractmethod
    def find_extreme_values(self,function_model,list_of_discontinuity_borders ):
     """Interface Method"""
    def get_maximum(self)-> Extreme_value_with_coordinates:
     """Interface Method"""    
    def get_minimum(self)-> Extreme_value_with_coordinates:
     """Interface Method"""
    def get_predictions(self):
     """Interface Method"""

class Extreme_value_setter_empty(IExtreme_value_setter):  # when using already normalized kernels
    def __init__(self):
        pass

    def find_extreme_values(self,function_model,list_of_discontinuity_borders ):
        pass

    def get_maximum(self)-> Extreme_value_with_coordinates:
        return None
    
    def get_minimum(self)-> Extreme_value_with_coordinates:
        return None

    def get_predictions(self):
        predictions_list = None
        return predictions_list
    
class Extreme_value_setter_solo_dimensionmax(IExtreme_value_setter): 
    def __init__(self, resolution=1000):
       self.resolution = resolution

    def find_extreme_values(self,list_of_discontinuity_borders, function_model:IGaussianProcessModel, ):
        self._locate_extreme_values(function_model=function_model,list_of_discontinuity_borders=list_of_discontinuity_borders)

    def get_maximum(self)-> Extreme_value_with_coordinates:
        maximum = self.maximum
        return maximum
    
    def get_minimum(self)-> Extreme_value_with_coordinates:
        minimum = self.minimum
        return minimum

    def get_predictions(self):
        predictions_list = self.predictions
        return predictions_list

    def _locate_extreme_values(self, list_of_discontinuity_borders,function_model:IGaussianProcessModel):
        dimensions= function_model.return_kernel_dimensions()
        model = function_model
        predictions_list_per_function = []
        coordinate_list_for_maximum_prediction = []
        coordinate_list_for_minimum_prediction = []
      
        for dim in list_of_discontinuity_borders: #go trough every dimension
            current_dimension = dim.dimension
            lower_border = dim.lower_border
            upper_border = dim.upper_border 
            reference_inputs_array_x = _inputloader.make_flat_array_for_x_inputs_as_reference(resolution=self.resolution, lower_bound=lower_border, upper_bound=upper_border)
            sparse_inputs_array_x = _inputloader.make_2d_array_for_x_inputs_sparse(dimensions=dimensions,filled_dimension=current_dimension ,resolution=self.resolution,lower_bound=lower_border ,upper_bound=upper_border)
            #predict values for solo_dimension_filled_input
            y_predictions = model.predict(sparse_inputs_array_x)
            y_predictions = y_predictions[0]
            y_predictions = y_predictions.flatten()
            #find coordinate of extreme_values for this dimension
            maximum_value = np.max(y_predictions)
            coordinate_for_max_value = self._find_the_coordinates_of_an_extremevalue(extreme_value=maximum_value,y_predictions=y_predictions, reference_inputs_array_x=reference_inputs_array_x)
            coordinate_list_for_maximum_prediction.append(coordinate_for_max_value)

            minimum_value = np.min( y_predictions )
            coordinate_for_min_value = self._find_the_coordinates_of_an_extremevalue(extreme_value=minimum_value,y_predictions=y_predictions, reference_inputs_array_x=reference_inputs_array_x)
            coordinate_list_for_minimum_prediction.append(coordinate_for_min_value)
            # save predictions for reuse
            prediction_object = Predictions(dimension=current_dimension,predicted_values=y_predictions, input_values=reference_inputs_array_x )
            predictions_list_per_function.append(prediction_object)

        #safe predictions
        self.predictions = predictions_list_per_function 
        #find and save actual extreme values
        self.maximum = self._make_an_extreme_value_object(model=model, coordinate_list_for_prediction= coordinate_list_for_maximum_prediction)
        self.minimum = self._make_an_extreme_value_object(model=model, coordinate_list_for_prediction= coordinate_list_for_minimum_prediction)

    def _make_an_extreme_value_object(self, coordinate_list_for_prediction,model)-> Extreme_value_with_coordinates:
        x_inputs_for_prediction = np.atleast_2d(coordinate_list_for_prediction)
        extreme_value = model.predict(x_inputs_for_prediction)
        extreme_value = extreme_value[0][0][0]
        extreme_value_object = Extreme_value_with_coordinates(value=extreme_value, coordinates= x_inputs_for_prediction)
        return extreme_value_object

    def _find_the_coordinates_of_an_extremevalue(self, extreme_value,y_predictions, reference_inputs_array_x)-> float:
            index_for_coordinates_of_extreme_value= np.where( y_predictions == extreme_value)
            coordinate_for_extreme_value= reference_inputs_array_x[index_for_coordinates_of_extreme_value]
            coordinate_for_extreme_value = coordinate_for_extreme_value[0]
            return coordinate_for_extreme_value
    

class INormalizer(metaclass=ABCMeta): # interface for strategies for setting extreme values currently only solo_dimension_max.   extreme_value_empty might be usable if extreme value setter is not needed due to already "normalized kernels"
    extreme_value_setter : IExtreme_value_setter
    @abstractmethod
    def normalize_value(self,input_value:float ):
     """Interface Method"""
    @abstractmethod
    def set_normalizer(self, output_minimum, output_maximum,input_max, input_min):
     """Interface Method"""
         
class Normalizer_minmax_stretch(INormalizer): 
    def __init__(self, extreme_value_setter: IExtreme_value_setter = Extreme_value_setter_solo_dimensionmax()):
        self.extreme_value_setter = extreme_value_setter
        self.output_minimum = 0
        self.output_maximum = 1

    def set_normalizer(self, output_minimum, output_maximum,input_max, input_min):
        self.output_minimum = output_minimum 
        self.output_maximum = output_maximum
        self.input_min = input_min
        self.input_max = input_max

    def normalize_value(self,input_value:float ):
        span_inputs = (self.input_max-self.input_min)
        if span_inputs == 0:
            normalized_value= (input_value-self.input_min)
            normalized_value= (input_value-self.input_min) + (self.output_maximum - self.output_minimum)/2
        else:
            squeeze_faktor = (self.output_maximum - self.output_minimum)/span_inputs
            normalized_value= (input_value-self.input_min)
            normalized_value = normalized_value * squeeze_faktor

        return normalized_value 

class Normalizer_minmax_stretch_random(INormalizer): 
    def __init__(self, extreme_value_setter: IExtreme_value_setter = Extreme_value_setter_solo_dimensionmax()):
        self.extreme_value_setter = extreme_value_setter
        self.output_minimum = 0
        self.output_maximum = 1

    def set_normalizer(self, output_minimum, output_maximum,input_max, input_min):
        self.output_minimum = output_minimum 
        self.output_maximum = output_maximum
        self.input_min = input_min
        self.input_max = input_max
        self.squeeze_faktor = (self.output_maximum - self.output_minimum)/(self.input_max-self.input_min)
        self.squeeze_corrector = random.uniform(0.1, 1) # this is the new squeeze value
        self.shift_corrector = random.uniform(0, (self.output_maximum-(self.output_maximum*self.squeeze_corrector))) # this is the new room for the value to "shift"

    def normalize_value(self,input_value:float ):
        span_inputs = (self.input_max-self.input_min)
        if span_inputs == 0:
            normalized_value= (input_value-self.input_min)
            normalized_value= (input_value-self.input_min) + self.shift_corrector
        else:
            self.squeeze_faktor = (self.output_maximum - self.output_minimum)/span_inputs
            squeez_corrector = self.squeeze_corrector
            shift_corrector = self.shift_corrector
            normalized_value= (input_value-self.input_min)
            normalized_value = normalized_value * self.squeeze_faktor
            normalized_value =normalized_value*squeez_corrector + shift_corrector
        return normalized_value 

class Normalizer_empty(INormalizer): 
    def __init__(self, extreme_value_setter= Extreme_value_setter_empty()):
        self.extreme_value_setter = extreme_value_setter

    def set_normalizer(self, output_minimum, output_maximum,input_max, input_min):
        pass

    def normalize_value(self,input_value:float ):
        return input_value