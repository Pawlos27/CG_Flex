
import numpy as np
import GPy
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any, List, Type, Tuple, Optional
import random
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import cgflex_project.Module_Dependencymaker._functionmaker_extreme_values as _functionmaker_extreme_values
import cgflex_project.Module_Dependencymaker._inputloader as _inputloader
import copy 
from  cgflex_project.Module_Dependencymaker._functionmaker_gaussian_process_models import IGaussianProcessModel, GPyModel

# dimension number always 0,1,2,3etc

def probabilitychecker(probability):
    return random.random() < probability

@dataclass
class discontinuity_borders_per_dimension:
    dimension: int
    lower_border : float
    upper_border :float

    def __post_init__(self):    
        pass

@dataclass
class Predictions_3d:
    resolution: int
    predictions : np.ndarray

    def __post_init__(self):    
        pass


class single_function_with_discontinuity:
    def __init__(self,function_model:IGaussianProcessModel,list_of_discontinuity_borders: List[discontinuity_borders_per_dimension],maximum_local: _functionmaker_extreme_values.Extreme_value_with_coordinates, minimum_local: _functionmaker_extreme_values.Extreme_value_with_coordinates, predictions: List[Type[_functionmaker_extreme_values.Predictions]]) :
        self.function_model = function_model
        self.list_of_discontinuity_borders = list_of_discontinuity_borders 
        self.maximum_local = maximum_local
        self.minimum_local = minimum_local
        self.predictions= predictions


class Dependency_functions:   # the class containing the dependency functions can predict/calculate the output and also corrects the input space
         
    def __init__(self, functions: List[single_function_with_discontinuity], normalizer: _functionmaker_extreme_values.INormalizer, range_of_output:Tuple[float,float]):
        self.functions = functions
        self.maximum_absolute = None
        self.minimum_absolute = None
        self.predictions_3d = None
        self.normalizer = normalizer
        self.range_of_output = range_of_output

    def calculate_value(self,x_inputs)->float:
        functions = self.functions
        model= self._extract_model_associated_with_inputs(functions= functions , inputs= x_inputs)
        x_inputs_formatted = np.array([x_inputs])
        predicted= model.predict(x_inputs_formatted)
        predicted_value=predicted[0][0][0]
        normalized_predicted_value = self.normalizer.normalize_value(input_value=predicted_value)
        return normalized_predicted_value
    
    def calculate_value_raw(self,x_inputs)->float:
        functions = self.functions
        model= self._extract_model_associated_with_inputs(functions= functions , inputs= x_inputs)
        x_inputs_formatted = np.array([x_inputs])
        predicted= model.predict(x_inputs_formatted)
        predicted_value=predicted[0][0][0]
        return predicted_value
    
    def make_predicitions_for_multiple_points_raw(self,x_inputs:List[List[float]])-> List[float]:
        predicted_value_list = []
        for inputs in x_inputs:
            predicted_value=self.calculate_value_raw(x_inputs=inputs)
            predicted_value_list.append(predicted_value)
        return predicted_value_list
    
    def make_predicitions_for_multiple_points(self,x_inputs:List[List[float]])-> List[float]:
        predicted_value_list = []
        for inputs in x_inputs:
            predicted_value=self.calculate_value(x_inputs=inputs)
            predicted_value_list.append(predicted_value)
        return predicted_value_list
    
    def _determine_and_set_absolute_extreme_values(self):# here 
        absolute_maximum = 0
        absolute_minimum = 0
        for function in self.functions:
            if function.maximum_local.value > absolute_maximum:
                absolute_maximum = function.maximum_local.value
            if function.minimum_local.value < absolute_minimum:
                absolute_minimum = function.minimum_local.value
        self.maximum_absolute = absolute_maximum
        self.minimum_absolute = absolute_minimum

    def set_normalizer(self,minimum_output,maximum_output): # always absolute stretching, possibility to stretch only if over the max, or to stretch randomly
        self._determine_and_set_absolute_extreme_values()
    
        self.normalizer.set_normalizer(output_minimum=minimum_output,output_maximum=maximum_output,input_min=self.minimum_absolute,input_max=self.maximum_absolute)

    def normalize_value(self, value:float):
        normalized_value = self.normalizer.normalize_value(input_value=value)
        return normalized_value
    
    def normalize_initial_prediction(self):
        pass
    
    def normalize_list_of_values(self, values:list):
        values_normalized = []
        for value in values:
            value_normalized = self.normalizer.normalize_value(input_value=value)
            values_normalized.append(value_normalized)
        return values_normalized
    
    def normalize_array_of_values(self, values:np.ndarray) ->np.ndarray:
        values_normalized = []

        for element in values:
            value_normalized = self.normalizer.normalize_value(input_value=element)
            values_normalized.append(value_normalized)

        values_normalized = np.array(values_normalized)

        return values_normalized

    def show_function_borders(self):
        functions = self.functions
        list_of_function_data = []
        function_counter=1
        for function in functions:
            model= function.function_model
            for dim in function.list_of_discontinuity_borders :
                dimension = dim.dimension
                lower_border = dim.lower_border 
                upper_border = dim.upper_border 
                function_border_data_dict = {"function_number":function_counter, "dimension":dimension, "lower_border": lower_border, "upper_border":upper_border}
                list_of_function_data.append(function_border_data_dict)
        df_function_with_borders =  pd.DataFrame(list_of_function_data)
        print(df_function_with_borders)

    def make_new_predictions_for_plotting(self, resolution):
   
        functions = self.functions
        function_counter=0
        resolution = resolution
        for function in functions:
            new_prediction_list = []
            model= function.function_model
            list_of_discontinuity_borders = function.list_of_discontinuity_borders 
            input_dim = model.return_kernel_dimensions()
            for dim in list_of_discontinuity_borders:
                dimension = dim.dimension
                lower_border = dim.lower_border 
                upper_border = dim.upper_border 
                new_resolution = (upper_border-lower_border)*resolution
                new_resolution = round(new_resolution)
                x_plot = _inputloader.make_flat_array_for_x_inputs_as_reference(resolution=resolution, lower_bound=lower_border, upper_bound=upper_border)
                x = _inputloader.make_2d_array_for_x_inputs_sparse(filled_dimension=dimension, resolution=resolution, lower_bound=lower_border, upper_bound=upper_border, dimensions=input_dim)
                predictions = model.predict(x)
                y_predictions =predictions[0]
                y_predictions = y_predictions.flatten()
                y_predictions = self.normalize_array_of_values(values=y_predictions)
                prediction_object = _functionmaker_extreme_values.Predictions(function_nr=function_counter,predicted_values=y_predictions,input_values=x_plot,dimension=dimension, normalized= True)
                new_prediction_list.append(prediction_object)
            function.predictions = new_prediction_list
            function_counter += 1

    def print_predictions(self):
        new_prediction_list = []
        for function in self.functions:
            predictionsliste = function.predictions
            for prediction in predictionsliste :
                new_prediction_list.append(prediction)
 
        for i in new_prediction_list:
                
                y_predictions = i.predicted_values
                x_plot = i.input_values
                function_nr = i.function_nr
                dimension = i.dimension
                output_data = pd.DataFrame({'function_number': function_nr,'dimension': dimension,'input_values': x_plot,"output_values": y_predictions})
                print(output_data)

    def show_functions_for_existing_predictions(self, label="unknown_function"):
        plt.figure()
        for function in self.functions:
            predictionsliste = function.predictions
            for prediction in predictionsliste :
                if prediction.normalized == False:
                    prediction.predicted_values = self.normalize_array_of_values(values=prediction.predicted_values)
                    prediction.normalized = True
                y_values = prediction.predicted_values
                x_values = prediction.input_values
                plt.plot(x_values, y_values)
        plt.xlabel('input per dimension')
        plt.ylabel('output')
        plt.title(label)
        plt.show()
        print("activated test")

    def make_new_3d_predictions(self, resolution=20):
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions >= 2:
    
            initial_plot =np.linspace(0, 1, resolution)
            y_values_for_x_combinations_list = []
            x = initial_plot
            y = initial_plot
            for i in initial_plot:
                sub_coordinate_list = []
                for j in initial_plot:
                    coordinate_point = [i, j]
                    sub_coordinate_list.append(coordinate_point)
                if dimensions >2:
                    for i in range(dimensions-2):
                        for coordinate in sub_coordinate_list:
                            coordinate.append(0)
                predicitions = self.make_predicitions_for_multiple_points(x_inputs=sub_coordinate_list)
                y_values_for_x_combinations_list.append(predicitions)
            z = np.atleast_2d(y_values_for_x_combinations_list)
            self.predictions_3d = Predictions_3d(resolution=resolution, predictions=z)

    def show_3d_plot(self, resolution=20, label="unknown_function"): # making 3d plot, when no plotdata existing or resolution unsufficient then calculating new data
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions >= 2:

            if self.predictions_3d == None:
                self.make_new_3d_predictions()
            elif self.predictions_3d.resolution < resolution:
                self.make_new_3d_predictions()
                
            resolution_final = self.predictions_3d.resolution
            z = self.predictions_3d.predictions
            x = np.linspace(0, 1, resolution_final)
            y = x
            X, Y = np.meshgrid(x,y)
            fig = plt.figure()
            ax = fig.add_subplot(111,projection="3d")
            ax.plot_surface(X, Y, z)
            ax.set_xlabel("input first dimension")
            ax.set_ylabel("input second dimension")
            ax.set_zlabel("output")
            plt.title(label)
            plt.show()

    def show_3d_plot_not_normalized(self, resolution=20, label="not_normalized_plot"):
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions >= 2:
    
            initial_plot =np.linspace(0, 1, resolution)
            y_values_for_x_combinations_list = []
            x = initial_plot
            y = initial_plot
            for i in initial_plot:
                sub_coordinate_list = []
                for j in initial_plot:
                    coordinate_point = [i, j]
                    sub_coordinate_list.append(coordinate_point)
                if dimensions >2:
                    for i in range(dimensions-2):
                        for coordinate in sub_coordinate_list:
                            coordinate.append(0)
                predicitions = self.make_predicitions_for_multiple_points_raw(x_inputs=sub_coordinate_list)
                y_values_for_x_combinations_list.append(predicitions)
                print(sub_coordinate_list)
            z = np.atleast_2d(y_values_for_x_combinations_list)
            x = np.linspace(0, 1, resolution)
            y = x
            X, Y = np.meshgrid(x,y)
            fig = plt.figure()
            ax = fig.add_subplot(111,projection="3d")
            ax.plot_surface(X, Y, z)
            ax.set_xlabel("x-axis")
            ax.set_ylabel("x2-axis")
            ax.set_zlabel("y-axis")
            plt.title(label=label)
            plt.show()

    def show_functions_3d_plot_for_first_two_dimensions_when_possible(self,resolution, label="unknown function"):
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions == 1:
            if self.functions[0].predictions == None:
                self.make_new_predictions_for_plotting(resolution=resolution)
            
            elif len(self.functions[0].predictions[0].predicted_values) < resolution :
                self.make_new_predictions_for_plotting(resolution=resolution)
            else:
                pass
            self.show_functions_for_existing_predictions(label=label)
        elif dimensions > 1 :
            self.show_3d_plot(resolution=resolution, label=label)
        else:
            pass

    def show_functions_3d_plot_if_exactly_two_dimensions(self,resolution,label="unknown_function"):
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions == 2:
            self.show_3d_plot(resolution=resolution,label=label)

        else:
            if self.functions[0].predictions == None:
                self.make_new_predictions_for_plotting(resolution=resolution)
            
            elif len(self.functions[0].predictions[0].predicted_values) < resolution :
                self.make_new_predictions_for_plotting(resolution=resolution)
            else:
                pass
            self.show_functions_for_existing_predictions(label= label)
  
    
    def _extract_model_associated_with_inputs(self,functions: List[Type[single_function_with_discontinuity]], inputs:list):
        model = functions[0].function_model
        miss = False
        if len(functions) > 1:
            for function in functions:
                miss = False
                for dim in function.list_of_discontinuity_borders:
                    dimensions_f = dim.dimension
                    if inputs[dimensions_f] < dim.lower_border or inputs[dimensions_f] > dim.upper_border :
    
                        miss = True
                        break
                if miss == False:
                    model = function.function_model
                    break                   
            if miss == True:
                for function in functions:
                    miss = False
                    for dim in function.list_of_discontinuity_borders:
                        dimensions_f = dim.dimension
                        # values might be out of the present ranges, thats why we have to set the edge borders to infinity
                        if dim.lower_border == self.range_of_output[0]:
                            lower_border_check = float('-inf')
                        else :
                            lower_border_check = dim.lower_border 
                        if dim.upper_border == self.range_of_output[1]:
                            upper_border_check = float('inf')
                        else :
                            upper_border_check = dim.upper_border 

                        if inputs[dimensions_f] < lower_border_check or inputs[dimensions_f] > upper_border_check:
                            miss = True
                            break
                    if miss == False:
                        model = function.function_model
                        break       
            if miss == True:
                print(f"funktion immernoch nicht zugeordner")
        return model



class IFunction_maker(metaclass=ABCMeta):
    @abstractmethod
    def load_inputs(self, dimensions, lower, upper):
     """Interface Method"""
    def make_functions(self, kernel, errorterm_tolerance:float,  range_of_output: Tuple[float, float]):
     """Interface Method"""
    


class Function_maker_evenly_discontinuity_in_one_dimension(IFunction_maker): # new discontinuity borders can appear only in the same dimensions used in the first discontinuity 
        
    def __init__(self, discontinuity_frequency:float = 0.2 ,  maximum_discontinuities:int = 2, discontinuity_reappearance_frequency:float = 0.4, extreme_value_setter: _functionmaker_extreme_values.IExtreme_value_setter = _functionmaker_extreme_values.Extreme_value_setter_solo_dimensionmax(resolution=100), normalizer:_functionmaker_extreme_values.INormalizer = _functionmaker_extreme_values.Normalizer_minmax_stretch(), gp_model= GPyModel(),inputloader= _inputloader.Inputloader_for_solo_random_values()):
        self.discontinuity_frequency = discontinuity_frequency
        self.maximum_discontinuities = maximum_discontinuities
        self.discontinuity_reappearance_frequency  = discontinuity_reappearance_frequency
        self.extreme_value_setter = extreme_value_setter
        self.normalizer = normalizer
        self.gp_model= gp_model
        self.inputloader = inputloader
        


    def load_inputs(self, dimensions, lower, upper):
        self.inputloader.set_input_loader(dimensions=dimensions, lower=lower, upper=upper)
        self.training_input_x= self.inputloader.load_x_training_data()
        self.training_input_y= self.inputloader.load_y_training_data()

    def make_functions(self, kernel, errorterm_tolerance:float,   range_of_output: Tuple[float, float])-> Dependency_functions :
        print("functionamker initialized")
        minimum_output= range_of_output[0]
        maximum_output= range_of_output[0]+((range_of_output[1]-range_of_output[0])*(1-errorterm_tolerance))
        dimensions = kernel.input_dim
        self.load_inputs(dimensions=dimensions, upper=minimum_output, lower=maximum_output)

        list_of_function_objects = []
        model = self.gp_model
        model.train(X=self.training_input_x,Y=self.training_input_y ,kernel=kernel)
        
        number_of_functions, list_of_dimensions_with_discontinuity = self._generate_number_for_functions_and_list_of_discontinuity_dimensions(dimensions=dimensions,discontinuity_frequency=self.discontinuity_frequency , discontinuity_reappearance_frequency=self.discontinuity_reappearance_frequency, maximum_discontinuities=self.maximum_discontinuities)
        list_of_borders_and_its_dimensionality = self._generate_list_of_borders_and_its_dimensionality(number_of_functions=number_of_functions,list_of_dimensions_with_discontinuity=list_of_dimensions_with_discontinuity,range_of_output=range_of_output)  #list_of_borders_and_its_dimensionality[i][0] for dimension and [i][1] for borderlist
        #sample_input_x = _inputloader.make_2d_array_for_x_inputs_full(dimensions=dimensions)
        sample_input_x = _inputloader.make_grid_for_hypercube(dimensions=dimensions, resolution=4)
        for j in range(number_of_functions):
            sample_function = self._sample_and_correct(model=model,sample_input_x=sample_input_x)
            function_model = copy.deepcopy(self.gp_model)
            function_model.train(X=sample_input_x,Y=sample_function,kernel=kernel)
            list_of_discontinuity_borders = []
    
            for i in range(dimensions):
                dimension = i
                discontinuity_borders_object = discontinuity_borders_per_dimension(dimension=i, lower_border=range_of_output[0], upper_border=range_of_output[1])
                list_of_discontinuity_borders.append(discontinuity_borders_object )
            for entry in list_of_borders_and_its_dimensionality:
                dimension=entry[0]
                list_of_discontinuity_borders[dimension].lower_border = entry[1][j]
                list_of_discontinuity_borders[dimension].upper_border= entry[1][j+1]
            self.extreme_value_setter.find_extreme_values( function_model=function_model,list_of_discontinuity_borders=list_of_discontinuity_borders )
            function_object= single_function_with_discontinuity(function_model=function_model, list_of_discontinuity_borders= list_of_discontinuity_borders, maximum_local=self.extreme_value_setter.get_maximum(), minimum_local=self.extreme_value_setter.get_minimum(), predictions=self.extreme_value_setter.get_predictions())
            list_of_function_objects.append(function_object)
        normalizer = copy.deepcopy(self.normalizer)

        dependency_functions = Dependency_functions(functions=list_of_function_objects, normalizer = normalizer, range_of_output=range_of_output )
        dependency_functions.set_normalizer(minimum_output=minimum_output,maximum_output=maximum_output)
        return dependency_functions
    
    @staticmethod
    def _sample_and_correct(sample_input_x, model:IGaussianProcessModel, size=1):
        sample = model.posterior_samples_f(X=sample_input_x, size=size)
        sample = sample.flatten()
        sample = sample[:, np.newaxis] 
        return sample

    @staticmethod
    def _generate_list_of_borders_and_its_dimensionality(number_of_functions:int, list_of_dimensions_with_discontinuity:list, range_of_output: Tuple[float, float]):

        list_of_borders_and_its_dimensionality=[]
        for i in list_of_dimensions_with_discontinuity:
            list_of_one_dimensions_and_its_borders = [i]
            list_of_borders=[range_of_output[0],range_of_output[1]]
            for j in range(number_of_functions-1):
                x= random.uniform(range_of_output[0], range_of_output[1])
                list_of_borders.append(x)
            list_of_borders.sort()
            list_of_one_dimensions_and_its_borders.append(list_of_borders)
            list_of_borders_and_its_dimensionality.append(list_of_one_dimensions_and_its_borders)
        return list_of_borders_and_its_dimensionality

    @staticmethod
    def _generate_number_for_functions_and_list_of_discontinuity_dimensions(dimensions:int, discontinuity_frequency:float, discontinuity_reappearance_frequency:float, maximum_discontinuities:int,maximum_number_of_dimensions_for_discontinuities=1):
        functions_counter = 1
        list_of_dimensions_with_discontinuity = []
        discontinuity= False
        discontinuity_dimensions_counter = 0

        if maximum_discontinuities > 0 :
            for j in range(dimensions):
                discontinuity_check = probabilitychecker(probability=discontinuity_frequency)
                if discontinuity_check == True :
                    discontinuity = True
                    list_of_dimensions_with_discontinuity.append(j)
                    discontinuity_dimensions_counter += 1
                if discontinuity_dimensions_counter >=maximum_number_of_dimensions_for_discontinuities:
                    break

        if discontinuity == True:  
            functions_counter += 1
        while functions_counter <= maximum_discontinuities and discontinuity == True :
            discontinuity_check = probabilitychecker(probability=discontinuity_reappearance_frequency)
            if discontinuity_check == True :
                functions_counter += 1
            else:
                discontinuity = False
        return functions_counter, list_of_dimensions_with_discontinuity



if __name__ == "__main__":

    import _kernelcombinator
    import _kernel_collection
    dimensions= 3
    kernel_kombination_maker= _kernelcombinator.Kernelcombinator_random_picking( kernel_operator_collection=_kernel_collection.Kernel__operator_collection_default(), kernel_selector= _kernelcombinator.Kernel_selector_random(max_dimensions_per_kernel= 1,kernel_collection= _kernel_collection.Kernel_collection_general_default()))

    ker2 = kernel_kombination_maker.combinate_kernels(dimensions=dimensions)

    ker = GPy.kern.RBF(input_dim=1, active_dims=[0],lengthscale=0.1) 

    ker = GPy.kern.RBF(input_dim=1, active_dims=[0],lengthscale=0.1) + GPy.kern.RBF(input_dim=1, active_dims=[1],lengthscale=0.2) 
    #GPy.kern.Linear(input_dim=2, active_dims=[1,2])
    inputlist= [0.1,0.1]



    function_maker = Function_maker_evenly_discontinuity_in_one_dimension()
    dependency_functions_object = function_maker.make_functions(kernel=ker, errorterm_tolerance=0.1,   range_of_output=(0,1))
    dependency_functions_object.show_function_borders()
    #value= dependency_functions_object.calculate_value(inputlist)
    #dependency_functions_object.make_predictions_show_and_print_functions(resolution=5)
    #dependency_functions_object.show_functions
    dependency_functions_object.show_functions_for_existing_predictions()
    #dependency_functions_object.print_predictions()
    dependency_functions_object.show_3d_plot()
    dependency_functions_object.show_3d_plot_not_normalized()
    dependency_functions_object.show_functions_3d_plot_for_first_two_dimensions_when_possible(resolution=20)
    dependency_functions_object.make_new_predictions_for_plotting(resolution=30)


    maximum_absolute = dependency_functions_object.maximum_absolute 
    max_abs_normalizer = dependency_functions_object.normalizer.input_max
    minimum_absolute = dependency_functions_object.minimum_absolute
    min_abs_normalizer = dependency_functions_object.normalizer.input_min
    min_soll = dependency_functions_object.normalizer.output_minimum
    max_soll = dependency_functions_object.normalizer.output_maximum
        
    print(f"the input min ist: {minimum_absolute}")
    print(f"the input max ist: {maximum_absolute}")
    print(f"the input min in the normalizer is ist: {min_abs_normalizer}")
    print(f"the input max in the normalizer is ist: {max_abs_normalizer}")
    print(f"the output min in the normalizer is ist: {min_soll}")
    print(f"the output max in the normalizer is ist: {max_soll}")

    max_funtion1 = dependency_functions_object.functions[0].maximum_local.value
    coordinates_max_funct_1 = dependency_functions_object.functions[0].maximum_local.coordinates
    coordinates_max_funct_1 =  coordinates_max_funct_1[0]
    coordinates_max_funct_1 = coordinates_max_funct_1.tolist()
    print(coordinates_max_funct_1)
    predicition = dependency_functions_object.calculate_value(x_inputs=coordinates_max_funct_1)

    print(f"the max of the first function is  {max_funtion1}")
    print(f"the actual_ raw prediction for that coordinates gives us   {predicition}")
    print(f"the shift is ")
    print(f"the stretch is ")
    print(f"the corrected prediction for that coordinates gives us   {predicition}")


    dependency_functions_object.show_functions_for_existing_predictions()
    #dependency_functions_object.show_functions_3d_plot_if_exactly_two_dimensions(resolution=40)


    maximum_absolute = dependency_functions_object.maximum_absolute 
    print(f"the absolute max ist: {maximum_absolute}")
    







