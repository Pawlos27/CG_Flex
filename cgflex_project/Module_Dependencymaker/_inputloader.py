
"""this module  contains a set of functions and classes provides utilities for generating and loading input data for various GP models or for plots,
 they offer flexibility in creating input data arrays, either by generating them based on specific criteria or by loading external data."""
import numpy as np
import matplotlib.pyplot as plt


import random
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import matplotlib.pyplot as plt
import cgflex_project.Shared_Classes.distributions as distributions


def make_2d_array_for_x_inputs_sparse(dimensions,filled_dimension,resolution=1000,lower_bound=0,upper_bound=1):
    """
    Generates a 2D array for input data with only one dimension filled and the rest sparse (filled with zeros).

    Args:
        dimensions (int): Total number of dimensions.
        filled_dimension (int): Index of the dimension to be filled with data.
        resolution (int, optional): Number of points to generate. Defaults to 1000.
        lower_bound (int, optional): Lower bound of the data range. Defaults to 0.
        upper_bound (int, optional): Upper bound of the data range. Defaults to 1.

    Returns:
        np.ndarray: A 2D numpy array with one dimension filled and the rest sparse.
    """
    new_resolution = resolution*(upper_bound - lower_bound)
    new_resolution = round(new_resolution)
    if new_resolution < 2:
        new_resolution = 2
    x_full = np.linspace(lower_bound, upper_bound, new_resolution)[:, np.newaxis] 
    x_empty = np.zeros_like(x_full)

    list_to_cocatenate=[x_empty]*dimensions
    list_to_cocatenate[filled_dimension]=x_full
    
    x = np.concatenate(list_to_cocatenate, axis=1)
    return x

def make_flat_array_for_x_inputs_as_reference(resolution,lower_bound=0,upper_bound=1):
    """
    Creates a flat (1D) array of input values within a specified range.

    Args:
        resolution (int): The number of points to generate.
        lower_bound (int, optional): Lower bound of the range. Defaults to 0.
        upper_bound (int, optional): Upper bound of the range. Defaults to 1.

    Returns:
        np.ndarray: A 1D numpy array of input values.
    """
    new_resolution = resolution*(upper_bound - lower_bound)
    new_resolution = round(new_resolution)
    if new_resolution < 2:
        new_resolution = 2
    x = np.linspace(lower_bound, upper_bound, new_resolution)
    return x

def make_2d_array_for_x_inputs_full(dimensions,resolution=1000,lower_bound=0,upper_bound=1):
    """
    Generates a 2D array where each dimension is uniformly filled with data.

    Args:
        dimensions (int): Total number of dimensions.
        resolution (int, optional): Number of points in each dimension. Defaults to 1000.
        lower_bound (int, optional): Lower bound of the data range. Defaults to 0.
        upper_bound (int, optional): Upper bound of the data range. Defaults to 1.

    Returns:
        np.ndarray: A 2D numpy array with all dimensions uniformly filled.
    """
    new_resolution = resolution*(upper_bound - lower_bound)
    new_resolution = round(new_resolution)
    if new_resolution < 2:
        new_resolution = 2
    x_full = np.linspace(lower_bound, upper_bound, new_resolution)[:, np.newaxis] 
    list_to_cocatenate=[x_full]*dimensions
    x = np.concatenate(list_to_cocatenate, axis=1)
    return x

def make_2d_array_for_inputs_full_randomly_distributed(dimensions, distribution: distributions.IDistributions, number_of_points=1):
    """
    Creates a 2D array with values randomly distributed across multiple dimensions.

    Args:
        dimensions (int): The number of dimensions.
        distribution (distributions.IDistributions): The distribution to use for generating random values.
        number_of_points (int, optional): The number of points to generate. Defaults to 1.

    Returns:
        np.ndarray: A 2D numpy array with randomly distributed values.
    """
    list_to_cocatenate=[]
    for i in range (dimensions):
        y = np.random.uniform(low=0,high=1,size=number_of_points)[:, np.newaxis] 
        x = np.atleast_2d(y)
        list_to_cocatenate.append(x)
    x = np.concatenate(list_to_cocatenate, axis=1)
    return x


def make_grid_for_hypercube(dimensions, resolution, lower_bound=0, upper_bound=1):
    """
    Generates a grid of points uniformly distributed over a hypercube.

    Args:
        dimensions (int): The number of dimensions of the hypercube.
        resolution (int): The number of points along each dimension.
        lower_bound (float, optional): The lower bound of the hypercube. Defaults to 0.
        upper_bound (float, optional): The upper bound of the hypercube. Defaults to 1.

    Returns:
        np.ndarray: A numpy array representing points in the hypercube.
    """
    axes = np.linspace(lower_bound, upper_bound, resolution)
    grid = np.meshgrid(*[axes]*dimensions)  # grid for each dimension
    grid = np.array(grid).T.reshape(-1, dimensions)  # reshaping
    return grid


class IInputloader(metaclass=ABCMeta):
    """
    An abstract base class defining the interface for loading training input data, this data is then reused for training GP models to generate random functions.
    """
    @abstractmethod
    def load_x_training_data(self):
        """Loads the input features (X) for training."""
    
    @abstractmethod
    def set_input_loader(self,dimensions:int, lower:int, upper:int):
        """
        Sets up the input loader with specified dimensions and value range.

        Args:
            dimensions (int): Number of dimensions for the input data.
            lower (int): Lower bound for input values.
            upper (int): Upper bound for input values.
        """
    
    @abstractmethod
    def load_y_training_data(self):
        """Loads the target/output values (Y) for training."""
      
    
class Inputloader_for_solo_random_values(IInputloader):
    """Implements the Interface and functions to generate the data, focuses on generating a single datapoint for model training"""
     
    def __init__(self):
        pass
    
    def set_input_loader(self,dimensions:int, lower:int, upper:int):
        self.lower = lower
        self.upper = upper
        self.dimensions = dimensions

    def load_x_training_data(self):
        self.training_input_x = make_2d_array_for_inputs_full_randomly_distributed(dimensions=self.dimensions,distribution=distributions.Distribution_uniform(min=self.lower, max=self.upper) )
        return self.training_input_x
    
    def load_y_training_data(self):
        self.training_input_y = make_2d_array_for_inputs_full_randomly_distributed(dimensions=1,distribution=distributions.Distribution_uniform(min=self.lower, max=self.upper))
        return self.training_input_y
       

       
class Inputloader_load_external_inputs(IInputloader):
     
    def __init__(self):
        pass
    def set_input_loader(self,training_input_array_y: np.ndarray, training_input_array_x: np.ndarray):
        self.training_input_y = training_input_array_y
        self.training_input_x = training_input_array_x
    def load_x_training_data(self):
       return self.training_input_x

    def load_y_training_data(self):
       return self.training_input_y
