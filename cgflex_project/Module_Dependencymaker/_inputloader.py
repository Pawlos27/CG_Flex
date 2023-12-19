
import numpy as np
import matplotlib.pyplot as plt


import random
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import matplotlib.pyplot as plt
import cgflex_project.Shared_Classes.distributions as distributions


def make_2d_array_for_x_inputs_sparse(dimensions,filled_dimension,resolution=1000,lower_bound=0,upper_bound=1): # resolution lower limit=2 for drawing purposes
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
    new_resolution = resolution*(upper_bound - lower_bound)
    new_resolution = round(new_resolution)
    if new_resolution < 2:
        new_resolution = 2
    x = np.linspace(lower_bound, upper_bound, new_resolution)
    return x

def make_2d_array_for_x_inputs_full(dimensions,resolution=1000,lower_bound=0,upper_bound=1):
    new_resolution = resolution*(upper_bound - lower_bound)
    new_resolution = round(new_resolution)
    if new_resolution < 2:
        new_resolution = 2
    x_full = np.linspace(lower_bound, upper_bound, new_resolution)[:, np.newaxis] 
    list_to_cocatenate=[x_full]*dimensions
    x = np.concatenate(list_to_cocatenate, axis=1)
    return x

def make_2d_array_for_inputs_full_randomly_distributed(dimensions, distribution: distributions.IDistributions, number_of_points=1):
    list_to_cocatenate=[]
    for i in range (dimensions):
        y = np.random.uniform(low=0,high=1,size=number_of_points)[:, np.newaxis] 
        x = np.atleast_2d(y)
        list_to_cocatenate.append(x)
    x = np.concatenate(list_to_cocatenate, axis=1)
    return x


def make_grid_for_hypercube(dimensions, resolution, lower_bound=0, upper_bound=1):
    # Erzeugt eine gleichmäßige Verteilung von Punkten über den Hyperwürfel
    axes = np.linspace(lower_bound, upper_bound, resolution)
    grid = np.meshgrid(*[axes]*dimensions)  # Erstellt ein Gitter für jede Dimension
    grid = np.array(grid).T.reshape(-1, dimensions)  # Umformen in eine Liste von Punkten
    return grid


class IInputloader(metaclass=ABCMeta):
    @abstractmethod
    def load_x_training_data(self):
        """Interface Method"""
    
    @abstractmethod
    def set_input_loader(self,dimensions:int, lower:int, upper:int):
        """Interface Method"""
    
    @abstractmethod
    def load_y_training_data(self):
        """Interface Method"""
    
class Inputloader_for_solo_random_values(IInputloader):
     
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
