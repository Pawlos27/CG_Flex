from abc import ABCMeta, abstractstaticmethod, abstractmethod
import math
import random
from scipy.interpolate import RBFInterpolator
import numpy as np
import itertools
import GPy
import matplotlib.pyplot as plt
from typing import Any, List, Type, Union


import cgflex_project.Module_Dependencymaker._inputloader as _inputloader
import cgflex_project.Module_Dependencymaker._functionmaker as _functionmaker
from cgflex_project.Shared_Classes.distributions import *
from  cgflex_project.Module_Dependencymaker._inputloader import make_grid_for_hypercube
from copy import deepcopy


        


class IInterpolation_model(metaclass=ABCMeta):
    """
    Interface for creating interpolation models, essential in dynamic error term modeling. Every Component of a mixture distribution gets an own interpolation model"""

    @abstractstaticmethod
    def return_number_of_required_values( dimensions:int)->float:
        """Determines the number of training data points required based on dimensions."""
    @abstractmethod
    def set_interpolator_model(self,values:List[float], dimensions:int):
        """Initializes and trains the interpolation model with provided values."""

    @abstractmethod
    def calculate_interpolated_values(self, inputs): 
        """Calculates interpolated values based on inputs.
 
        """

    def plot_interpolator_model(self, label="interpolation of one component of errorterm"):
        dimensions = self.data_points.shape[1]
        if dimensions == 1:
            self._plot_1d(label=label + f"interpolation of one component of errorterm")
        elif dimensions == 2:
            self._plot_2d(label=label + f"interpolation of one component of errorterm")
        elif dimensions > 2:
            self._plot_multidim(label=label + f"interpolation of one component of errorterm")   
    
    def _plot_1d(self, label):
        """ For 1D, we use a simple line plot"""
        x = np.linspace(np.min(self.data_points), np.max(self.data_points), 200)
        y = self.calculate_interpolated_values(inputs=x[:, np.newaxis])
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data_points, self.values, color='red', label='Original Data Points')
        plt.plot(x, y, color='blue', label='Interpolated Values')
        plt.title(label)
        plt.xlabel('X-axis')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
        
    
    def _plot_2d(self, label):
        """ For 2D, we use 3d plot"""
        # For 2D, proceed as before
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        grid_x, grid_y = np.meshgrid(x, y)
        shaped_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        # interpolation at the grid points
        interpolated_values = self.calculate_interpolated_values(inputs=shaped_points)
        Z = interpolated_values.reshape(grid_x.shape)

        # 3D scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data_points[:, 0], self.data_points[:, 1], self.values, color='red', label='Original Data Points')
        ax.scatter(grid_x.ravel(), grid_y.ravel(), Z.ravel(), color='blue', alpha=0.1, label='Interpolated Grid Points')
       
       #adjust 
        min_value = np.min(self.values)
        max_value = np.max(self.values)
        ax.set_zlim(min_value, max_value)
       
        ax.set_xlabel('inputs first dimension')
        ax.set_ylabel('inputs second dimension')
        ax.set_zlabel('Interpolated Values')
        ax.legend()
        plt.title(label)
        plt.show()
   
    def _plot_multidim(self, label):
    
        """for multidimensions >2 we just plot the first 2 dimensions, the rest dimensions are always set to 0
        """
        x = np.linspace(np.min(self.data_points[:, 0]), np.max(self.data_points[:, 0]), 50)
        y = np.linspace(np.min(self.data_points[:, 1]), np.max(self.data_points[:, 1]), 50)
        grid_x, grid_y = np.meshgrid(x, y)

        # Extend grid points to match dimensions of data points
        extended_grid_points = np.zeros((grid_x.size, self.dimensions))
        extended_grid_points[:, 0] = grid_x.ravel()
        extended_grid_points[:, 1] = grid_y.ravel()

        # Perform the interpolation at the grid points
        interpolated_values = self.calculate_interpolated_values(inputs=extended_grid_points)
        Z = interpolated_values.reshape(grid_x.shape)

        # 3D scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data_points[:, 0], self.data_points[:, 1], self.values, color='red', label='Original Data Points')
        ax.scatter(grid_x.ravel(), grid_y.ravel(), Z.ravel(), color='blue', alpha=0.1, label='Interpolated Grid Points')
        
        #adjust 
        min_value = np.min(self.values)
        max_value = np.max(self.values)
        ax.set_zlim(min_value, max_value)
       
       #adjust 
        min_value = np.min(self.values)
        max_value = np.max(self.values)
        ax.set_zlim(min_value, max_value)
       
        ax.set_xlabel('inputs first dimension')
        ax.set_ylabel('inputs second dimension')
        ax.set_zlabel('Interpolated Values')
        ax.legend()
        plt.title(label)
        plt.show()



class Interpolation_model_RBFInterpolator_datapoints_grid(IInterpolation_model): # we need controlls for the string used to set the kernel
    """
    Uses RBF Interpolator from scipy with data points distributed over a hypercube grid.

    Arguments:
    - interpolation_kernel: Defines the kernel type for interpolation (e.g., "linear", "gaussian").

    Suitable for handling one-dimensional to multi-dimensional interpolation with effective control and precision.
    """
    def __init__(self) :
        self.interpolation_kernel = "gaussian"

    def _set_interpolator(self):
        if self.interpolation_kernel == "linear":
            self.interpolator =RBFInterpolator(self.data_points, self.values, kernel= self.interpolation_kernel)
        elif self.interpolation_kernel == "gaussian":
            self.interpolator =RBFInterpolator(self.data_points, self.values, kernel= self.interpolation_kernel, epsilon=5)

    def calculate_interpolated_values(self, inputs):
        interpolated_values = self.interpolator(inputs)
        return interpolated_values

    @staticmethod
    def return_number_of_required_values( dimensions:int):
        number = 3**dimensions 
        return number
   
    def set_interpolator_model(self,values:List[float], dimensions:int, range=(0,1)):
        self.dimensions = dimensions
        self.values = np.array(values)
        self._set_data_points(range=range)
        self._set_interpolator()

    def _set_data_points(self, range:tuple):
        """Generates data points that are spread on the hypercube .
        """
        #edge_points = np.array([np.array(i) for i in itertools.product([0, 1], repeat=self.dimensions)])
        #random_points = np.random.rand(self.dimensions**2, self.dimensions)
    
        #self.data_points = np.vstack([edge_points, random_points])
        data_points = make_grid_for_hypercube(dimensions=self.dimensions,resolution=3,lower_bound=range[0], upper_bound=range[1])
        self.data_points = data_points


class Interpolation_model_RBFInterpolator_datapoints_random(IInterpolation_model):
    """
    Uses RBF Interpolator from scipy with data points distributed randomly, 
    worse control then Interpolation_model_RBFInterpolator_datapoints_grid, but can work with less datapoints, thus creating slimmer models

    Arguments:
    - interpolation_kernel: Defines the kernel type for interpolation (e.g., "linear", "gaussian").

    Suitable for handling one-dimensional to multi-dimensional interpolation.
    """
    def __init__(self) -> None:
        self.interpolation_kernel = "gaussian"

    def _set_interpolator(self):
        if self.interpolation_kernel == "linear":
            self.interpolator =RBFInterpolator(self.data_points, self.values, kernel= self.interpolation_kernel)
        elif self.interpolation_kernel == "gaussian":
            self.interpolator =RBFInterpolator(self.data_points, self.values, kernel= self.interpolation_kernel, epsilon=5)

    def calculate_interpolated_values(self, inputs):
        interpolated_values = self.interpolator(inputs)
        return interpolated_values

    @staticmethod
    def return_number_of_required_values( dimensions:int):
        number = np.random.random_integers(low=2,high=10)
        number = number* dimensions
        return number
   
    def set_interpolator_model(self,values:List[float], dimensions:int):
        self.dimensions = dimensions
        self.values = np.array(values)
        self._set_data_points(num_data_points= len(values))
        self._set_interpolator()
   
    def _set_data_points(self, num_data_points:int):
        """Generates data points that equally spaced along the cross profile of the space
        """
        if num_data_points < 2:
            raise ValueError("the interpolator needs at least 2 values to interpolate")
        random_points = np.random.rand(num_data_points-2, self.dimensions)

        # Define edge points
        edge_point_low = np.zeros((1, self.dimensions))  # [0, 0, ..., 0]
        edge_point_high = np.ones((1, self.dimensions))  # [1, 1, ..., 1]

        # Combine random points with edge points
        self.data_points = np.vstack([edge_point_low, random_points, edge_point_high])



if __name__ == "__main__":
  
    # interpolation demonstration

    interpolator = Interpolation_model_RBFInterpolator_datapoints_grid()
    number = interpolator.return_number_of_required_values(dimensions=2)
    print(number)
    list_interpolation_models = []
    mus_listen_maker = Complex_distribution_list_of_Mus_maker(number_of_maximum_modes=2, total_number_of_elements=10,maximum_distance_between_modes_factor=5)
    mu=mus_listen_maker.sample_lists_of_mus(maximum_total_deviation=0.5,size_of_samples=number)
    sigma= mus_listen_maker.return_sigma()
    counter = 1
    for liste in mu :
        distribution = Distribution_mixture_of_normals(list_of_mus=liste, sigma=sigma)
        distribution.plot_distribution(label= f"mixture no: {counter}    " )
        counter += 1
    for i in range(10):
        values_train = []
        for liste in mu :
            values_train.append(liste[i])
        interpolator = interpolator
        interpolator2 = deepcopy(interpolator)
        interpolator2.set_interpolator_model(values=values_train, dimensions=2)
        list_interpolation_models.append(interpolator2)
        if i == 9:
            interpolator2.plot_interpolator_model(label= f"interpolation {i} component of mixtures")
        


    #mu = mu[0]
    #sigma= mus_listen_maker.return_sigma()

    #distribution = Distribution_mixture_of_normals(list_of_mus=mu, sigma=sigma)
    #distribution.plot_distribution(label= "bimodal complex -  close modes    " )
                                    
