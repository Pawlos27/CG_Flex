
import numpy as np
import GPy
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any, List, Type, Tuple, Optional, Union
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
    """
    Checks if a randomly generated number is less than the specified probability.

    Args:
        probability (float): The probability threshold to check against.

    Returns:
        bool: True if the random number is less than the probability, False otherwise.
    """
    return random.random() < probability

@dataclass
class discontinuity_borders_per_dimension:
    """
    Data class representing borders for a partial space for a specific dimension.
    Functions have assigned subregions for which they are valid, allowing for the implementation of discontinuity..

    Attributes:
        dimension (int): The specific dimension the borders apply to.
        lower_border (float): The lower boundary value of the discontinuity.
        upper_border (float): The upper boundary value of the discontinuity.
    """
    dimension: int
    lower_border : float
    upper_border :float

    def __post_init__(self):    
        pass

@dataclass
class Predictions_3d:
    """
    Data class for storing 3D predictions, used for visualisation of functions.
    """
    resolution: int
    """The resolution of the predictions."""

    predictions: np.ndarray
    """The array of predicted values."""

    visualized_dimensions: Tuple[int, int] = (0, 1)
    """The dimensions visualized in the predictions."""

    def __post_init__(self):    
        pass


class single_function_with_discontinuity:
    """
    Class representing a single function with specified discontinuities.

    Attributes:
        function_model (IGaussianProcessModel): The Gaussian process model for the function.
        list_of_discontinuity_borders (List[discontinuity_borders_per_dimension]): 
            A list of discontinuity borders for different dimensions, representing the input space for which the function is valid.
    """
    def __init__(self,function_model:IGaussianProcessModel,list_of_discontinuity_borders: List[discontinuity_borders_per_dimension]) :
        self.function_model = function_model
        self.list_of_discontinuity_borders = list_of_discontinuity_borders 

    def set_extreme_values(self, maximum_local: _functionmaker_extreme_values.Extreme_value_with_coordinates, minimum_local: _functionmaker_extreme_values.Extreme_value_with_coordinates, predictions: Union[None, List[_functionmaker_extreme_values.Predictions]]):
        """
        Sets the extreme values for the function model, as a byprocess predictions for the valid input space can be added .

        Args:
            maximum_local (_functionmaker_extreme_values.Extreme_value_with_coordinates): 
                The local maximum value with its coordinates.
            minimum_local (_functionmaker_extreme_values.Extreme_value_with_coordinates): 
                The local minimum value with its coordinates.
            predictions (Union[None, List[_functionmaker_extreme_values.Predictions]]): 
                The list of predictions associated with the function.
        """
        self.maximum_local = maximum_local
        self.minimum_local = minimum_local
        self.predictions= predictions


class Dependency_functions: 
    """
    Class containing the dependency functions that can predict/calculate the output and also correct the input space, it is the main class to manage the determenistic function model, the calculation, normalisation and visualisation of values.

    Attributes:
        functions (List[single_function_with_discontinuity]): List of functions and their valid inputspace.
        normalizer (_functionmaker_extreme_values.INormalizer): Normalizer used for value normalization.
        range_of_output (Tuple[float, float]): The range of the output values.
    """
         
    def __init__(self, functions: List[single_function_with_discontinuity], normalizer: _functionmaker_extreme_values.INormalizer, range_of_output:Tuple[float,float]):
        self.functions = functions
        self.maximum_absolute = None
        self.minimum_absolute = None
        self.predictions_3d = None
        self.normalizer = normalizer
        self.range_of_output = range_of_output

    def calculate_value(self,x_inputs)->float:
        """
        Calculates the normalized output value for given input values.

        Args:
            x_inputs (list): List of input values.

        Returns:
            float: The normalized output value.
        """
        functions = self.functions
        model= self._extract_model_associated_with_inputs(functions= functions , inputs= x_inputs)
        x_inputs_formatted = np.array([x_inputs])
        predicted= model.predict(x_inputs_formatted)
        predicted_value=predicted[0][0][0]
        normalized_predicted_value = self.normalizer.normalize_value(input_value=predicted_value)
        return normalized_predicted_value
    
    def calculate_value_raw(self,x_inputs)->float:
        """
        Calculates the raw output value without normalization for given input values.

        Args:
            x_inputs (list): List of input values.

        Returns:
            float: The raw output value.
        """
        functions = self.functions
        model= self._extract_model_associated_with_inputs(functions= functions , inputs= x_inputs)
        x_inputs_formatted = np.array([x_inputs])
        predicted= model.predict(x_inputs_formatted)
        predicted_value=predicted[0][0][0]
        return predicted_value
    
    def make_predicitions_for_multiple_points_raw(self,x_inputs:List[List[float]])-> List[float]:
        """
        Makes raw predictions for multiple input points without normalization.

        Args:
            x_inputs (List[List[float]]): A list of lists, each inner list containing input values.

        Returns:
            List[float]: A list of predicted raw output values.
        """
        predicted_value_list = []
        for inputs in x_inputs:
            predicted_value=self.calculate_value_raw(x_inputs=inputs)
            predicted_value_list.append(predicted_value)
        return predicted_value_list
    
    def make_predicitions_for_multiple_points(self,x_inputs:List[List[float]])-> List[float]:
        """
        Makes normalized predictions for multiple input points.

        Args:
            x_inputs (List[List[float]]): A list of lists, each inner list containing input values.

        Returns:
            List[float]: A list of predicted normalized output values.
        """
        predicted_value_list = []
        for inputs in x_inputs:
            predicted_value=self.calculate_value(x_inputs=inputs)
            predicted_value_list.append(predicted_value)
        return predicted_value_list

    def set_normalizer(self,minimum_output,maximum_output):
        """
        Sets the normalizer parameters based on the found extreme values.

        Args:
            minimum_output (float): The minimum value of the output range.
            maximum_output (float): The maximum value of the output range.
        """ 
        self._find_and_set_extreme_values()
        self.normalizer.set_normalizer(output_minimum=minimum_output,output_maximum=maximum_output,input_min=self.minimum_absolute,input_max=self.maximum_absolute)

    def normalize_value(self, value:float):
        """
        Normalizes a single value.

        Args:
            value (float): The value to normalize.

        Returns:
            float: The normalized value.
        """
        normalized_value = self.normalizer.normalize_value(input_value=value)
        return normalized_value
    
    
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

    def show_function_borders(self, node_id= "unknown"):
        """
        Prints an overview of responsible functions based on input ranges for each dimension.

        Args:
            node_id (str, optional): Identifier for the node. Defaults to "unknown".
        """

        functions = self.functions
        list_of_function_data = []
        function_counter=0
        for function in functions:
            model= function.function_model
            for dim in function.list_of_discontinuity_borders :
                dimension = dim.dimension
                lower_border = dim.lower_border 
                upper_border = dim.upper_border 
                function_border_data_dict = {"function_number":function_counter, "dimension":dimension, "lower_border": lower_border, "upper_border":upper_border}
                list_of_function_data.append(function_border_data_dict)
            function_counter +=1
        df_function_with_borders =  pd.DataFrame(list_of_function_data)
        print(f"Node_ID {node_id}   Overviev of responsible functions depending on the input range per dimension")
        print(df_function_with_borders)

    def make_new_predictions_for_plotting(self, resolution):
        """
        Generates new predictions for plotting purposes, for 2d plotting.

        Args:
            resolution (int): Resolution for plotting.
        """
   
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
        """
        Prints the predictions stored in the functions. Just for analytical purposes.
        """
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
        """
        Plot functions for existing predictions.

        Args:
            label (str, optional): Label for the plot. Defaults to "unknown_function".
        """
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

    def make_new_3d_predictions(self, resolution=20, visualized_dimensions: Tuple[int, int] = (0, 1)):
        """
        Generates new 3D predictions based on the vidualization of specified dimensions, rest of the dimensions stays at a fixed value of 0.

        Args:
            resolution (int, optional): Resolution for 3D plotting. Defaults to 20.
            visualized_dimensions (Tuple[int, int], optional): Dimensions to visualize in 3D plot. Defaults to (0, 1).
        """
        dimensions = self.functions[0].function_model.return_kernel_dimensions()
        if dimensions < 2:
            raise ValueError("No 3D visualisation of 1D data possible")
        elif max(visualized_dimensions) >= dimensions:
           visualized_dimensions = (0, 1)

        initial_plot = np.linspace(self.range_of_output[0], self.range_of_output[1], resolution)
        y_values_for_x_combinations_list = []

        for i in initial_plot:
            sub_coordinate_list = []
            for j in initial_plot:
                coordinate_point = [0] * dimensions
                coordinate_point[visualized_dimensions[0]] = i
                coordinate_point[visualized_dimensions[1]] = j
                sub_coordinate_list.append(coordinate_point)
            predicitions = self.make_predicitions_for_multiple_points(x_inputs=sub_coordinate_list)
            y_values_for_x_combinations_list.append(predicitions)

        z = np.atleast_2d(y_values_for_x_combinations_list)
        self.predictions_3d = Predictions_3d(resolution=resolution, predictions=z, visualized_dimensions=visualized_dimensions)

  
    def show_3d_plot(self, resolution=20, visualized_dimensions: Tuple[int, int] = (0, 1),label="unknown_function"):
        """
        Displays a 3D plot of the function based on provided dimensions and resolution. When no Predictions existing or resolution unsufficient then calculating new Predictions

        Args:
            resolution (int, optional): Resolution for 3D plotting. Defaults to 20.
            visualized_dimensions (Tuple[int, int], optional): Dimensions to visualize in 3D plot. Defaults to (0, 1).
            label (str, optional): Label for the plot. Defaults to "unknown_function".
        """ 
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions < 2:
            raise ValueError("No 3D visualisation of 1D data possible")
        elif max(visualized_dimensions) >= dimensions:
           visualized_dimensions = (0, 1)

        if self.predictions_3d == None:
            self.make_new_3d_predictions()
        elif self.predictions_3d.visualized_dimensions != visualized_dimensions :
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
        ax.plot_surface(Y, X, z)
        ax.set_xlabel(f"input dimension: {visualized_dimensions[0]}")
        ax.set_ylabel(f"input dimension: {visualized_dimensions[1]}")
        ax.set_zlabel("output")
        plt.title(label)
        plt.show()

    def show_3d_plot_not_normalized(self, resolution=20, label="not_normalized_plot"):
        """
        Displays a 3D plot of the raw (not normalized) function data.

        Args:
            resolution (int, optional): Resolution for 3D plotting. Defaults to 20.
            label (str, optional): Label for the plot. Defaults to "not_normalized_plot".
        """
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

    def show_functions_3d_plot_when_possible(self,resolution, visualized_dimensions: Tuple[int, int] = (0, 1),label="unknown function"):
        """
        Displays a 3D plot if the dimensions allow, otherwise shows existing predictions.

        Args:
            resolution (int): Resolution for the plot.
            visualized_dimensions (Tuple[int, int], optional): Dimensions to visualize in 3D plot. Defaults to (0, 1).
            label (str, optional): Label for the plot. Defaults to "unknown function".
        """
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions == 1:
            if self.functions[0].predictions == None:
                self.make_new_predictions_for_plotting(resolution=resolution)
            elif len(self.functions[0].predictions[0].predicted_values) < resolution :
                self.make_new_predictions_for_plotting(resolution=resolution)
            self.show_functions_for_existing_predictions(label=label)
        elif dimensions > 1 :
            self.show_3d_plot(resolution=resolution, label=label,visualized_dimensions=visualized_dimensions )


    def show_functions_3d_plot_if_exactly_two_dimensions(self,resolution,label="unknown_function"):
        """
        Displays a 3D plot specifically for functions with exactly two dimensions.

        Args:
            resolution (int): Resolution for the plot.
            label (str, optional): Label for the plot. Defaults to "unknown_function".
        """
        dimensions= self.functions[0].function_model.return_kernel_dimensions()
        if dimensions == 2:
            self.show_3d_plot(resolution=resolution,label=label)
        else:
            if self.functions[0].predictions == None:
                self.make_new_predictions_for_plotting(resolution=resolution)
            elif len(self.functions[0].predictions[0].predicted_values) < resolution :
                self.make_new_predictions_for_plotting(resolution=resolution)
            self.show_functions_for_existing_predictions(label= label)
  
    def _extract_model_associated_with_inputs(self,functions: List[Type[single_function_with_discontinuity]], inputs:list):
        """
        Extracts the model associated with the given inputs from a list of functions.

        Args:
            functions (List[single_function_with_discontinuity]): List of functions to search.
            inputs (list): List of input values.

        Returns:
            IGaussianProcessModel: The Gaussian process model associated with the inputs.
        """
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

    def _find_and_set_extreme_values(self):
        """
        Finds and sets the extreme values (maximum and minimum) for the dependency functions.
        """
        for single_function in self.functions:
            extreme_value_setter = self.normalizer.extreme_value_setter
            extreme_value_setter.find_extreme_values( function_model=single_function.function_model,list_of_discontinuity_borders=single_function.list_of_discontinuity_borders )
            single_function.set_extreme_values(maximum_local=extreme_value_setter.get_maximum(), minimum_local=extreme_value_setter.get_minimum(), predictions=extreme_value_setter.get_predictions())
        self._determine_and_set_absolute_extreme_values()
        if isinstance(self.normalizer.extreme_value_setter, _functionmaker_extreme_values.Extreme_value_setter_solo_dimensionmax):
            self.gridsearch_extreme_values()
        
    def _determine_and_set_absolute_extreme_values(self):
        """
        Determines and sets the absolute extreme values (maximum and minimum) across all functions.
        """
        absolute_maximum = self.functions[0].maximum_local.value
        absolute_minimum = self.functions[0].minimum_local.value
        for function in self.functions[1:]:
            if function.maximum_local.value > absolute_maximum:
                absolute_maximum = function.maximum_local.value
            if function.minimum_local.value < absolute_minimum:
                absolute_minimum = function.minimum_local.value
        self.maximum_absolute = absolute_maximum
        self.minimum_absolute = absolute_minimum

    def gridsearch_extreme_values(self, resolution=3):
        """
        Performs a grid search to find extreme values across all functions. This method to do a rough check of the actual used strategy.

        Args:
            resolution (int, optional): Resolution for the grid search. Defaults to 3.
        """
        check_inputs = _inputloader.make_grid_for_hypercube(dimensions=self.functions[0].function_model.return_kernel_dimensions(), lower_bound= self.range_of_output[0], upper_bound=self.range_of_output[1], resolution=resolution)
        check_inputs.tolist()
        predictions = self.make_predicitions_for_multiple_points_raw(x_inputs=check_inputs)
        max_prediction = max(predictions)
        min_prediction = min(predictions)
        if max_prediction > self.maximum_absolute:
            print(f"{max_prediction} is bigger {self.maximum_absolute}")
            self.maximum_absolute = max_prediction
        if min_prediction < self.minimum_absolute:
            print(f"{min_prediction} is smaller {self.minimum_absolute}")
            self.minimum_absolute = min_prediction



class IFunction_maker(metaclass=ABCMeta):
    """
    Abstract base class defining the interface for function makers. 
    Function makers are responsible for creating dependency functions based on specified criteria.
    Dependency_functions are a crucial class to represent dependencies of nodes from its parent nodes. 
    They represent a determenistic approach in modeling a function with gaussian processes.
    """
    @abstractmethod
    def load_inputs(self, dimensions, lower, upper):
     """
        Loads input data for generating/training the models for dependency functions.
        
        Args:
            dimensions (int): The number of dimensions for the input data.
            lower (float): The lower bound of the input data range.
            upper (float): The upper bound of the input data range.
        """
    def make_functions(self, kernel, errorterm_tolerance:float,  range_of_output: Tuple[float, float])-> Dependency_functions:
     """
        Creates and returns dependency functions based on the provided kernel and error term tolerance.
        
        Args:
            kernel: The kernel to be used in the Gaussian Process model.
            errorterm_tolerance (float): The tolerance level of the error term is taken into consideration in the output range.
            range_of_output (Tuple[float, float]): The range of output values.
        
        Returns:
            Dependency_functions(Dependency_functions): An object encapsulating the created dependency functions.
        """
    


class Function_maker_evenly_discontinuity_in_one_dimension(IFunction_maker): 
    """
    Implementation of the IFunction_maker interface. Generates functions with discontinuities
    evenly distributed in one dimension. The input space of one dimension is seperated into regions.

    First, it is determined whether there are any discontinuities and how many. If a discontinuity occurs, the input space in one dimension is divided into several regions.
    For each region, a separate Gaussian process model is generated, which represents a function, and then saved along with the information for the valid input ranges.
    When resolving the dependency later, it depends on the input value which function is responsible.

    Args:
            discontinuity_frequency (float): The frequency of introducing discontinuities.
            maximum_discontinuities (int): The maximum number of discontinuities allowed in one dependency representation.
            discontinuity_reappearance_frequency (float): The frequency of reappearing discontinuities in subsequent functions.
            sampling_resolution (int): The resolution of the sampling grid.
            normalizer (INormalizer): The normalizer to be used for value normalization.
            gp_model (IGaussianProcessModel): The Gaussian Process model to be used.
            inputloader (IInputloader): The input loader for loading training data.
    """
        
    def __init__(self, discontinuity_frequency:float = 0.2 ,  maximum_discontinuities:int = 2, discontinuity_reappearance_frequency:float = 0.4,sampling_resolution:int = 4, normalizer:_functionmaker_extreme_values.INormalizer = _functionmaker_extreme_values.Normalizer_minmax_stretch(), gp_model= GPyModel(),inputloader= _inputloader.Inputloader_for_solo_random_values()):
        self.discontinuity_frequency = discontinuity_frequency
        self.maximum_discontinuities = maximum_discontinuities
        self.discontinuity_reappearance_frequency  = discontinuity_reappearance_frequency
        self.normalizer = normalizer
        self.gp_model= gp_model
        self.inputloader = inputloader
        self.sampling_resolution = sampling_resolution

    def load_inputs(self, dimensions, lower, upper):
        self.inputloader.set_input_loader(dimensions=dimensions, lower=lower, upper=upper)
        self.training_input_x= self.inputloader.load_x_training_data()
        self.training_input_y= self.inputloader.load_y_training_data()

    def make_functions(self, kernel, errorterm_tolerance:float,   range_of_output: Tuple[float, float])-> Dependency_functions :
        minimum_output= range_of_output[0]
        maximum_output= range_of_output[0]+((range_of_output[1]-range_of_output[0])*(1-errorterm_tolerance))
        dimensions = kernel.input_dim
        self.load_inputs(dimensions=dimensions, upper=minimum_output, lower=maximum_output)
        list_of_function_objects = []
        model = self.gp_model
        model.train(X=self.training_input_x,Y=self.training_input_y ,kernel=kernel)
        number_of_functions, list_of_dimensions_with_discontinuity = self._generate_number_for_functions_and_list_of_discontinuity_dimensions(dimensions=dimensions,discontinuity_frequency=self.discontinuity_frequency , discontinuity_reappearance_frequency=self.discontinuity_reappearance_frequency, maximum_discontinuities=self.maximum_discontinuities)
        list_of_borders_and_its_dimensionality = self._generate_list_of_borders_and_its_dimensionality(number_of_functions=number_of_functions,list_of_dimensions_with_discontinuity=list_of_dimensions_with_discontinuity,range_of_output=range_of_output)  #list_of_borders_and_its_dimensionality[i][0] for dimension and [i][1] for borderlist
        sample_input_x = _inputloader.make_grid_for_hypercube(dimensions=dimensions, resolution=self.sampling_resolution)
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


            function_object= single_function_with_discontinuity(function_model=function_model, list_of_discontinuity_borders= list_of_discontinuity_borders)
            list_of_function_objects.append(function_object)
        normalizer = copy.deepcopy(self.normalizer)

        dependency_functions = Dependency_functions(functions=list_of_function_objects, normalizer = normalizer, range_of_output=range_of_output)
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
    dependency_functions_object.show_functions_3d_plot_when_possible(resolution=20)
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
    







