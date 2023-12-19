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



class IErrorterm_mixture_distance_distribution_collection(metaclass=ABCMeta):
    @abstractmethod
    def get_distance_array(self, size:int,maximum_distance:float ):
        """Interface Method"""


class Errorterm_mixture_distance_distribution_random(IErrorterm_mixture_distance_distribution_collection):
    def __init__(self):
        self.distribution_list=[Distribution_uniform(min=0, max=1),Distribution_mixture_of_normals_truncated_custom(list_of_mus=[1], sigma=0.00),Distribution_uniform(min=0.3, max=0.6),Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.1], sigma=0.1,lower_border=0.25, upper_border=0.55),Distribution_uniform(min=0.1, max=0.1),Distribution_uniform(min=0.9, max=1),Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.9,1,1.1,], sigma=0.7), Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.9,1,1.1,], sigma=0.2),Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.4], sigma=1)] 
    def get_distance_array(self, size,maximum_distance):
       random_distribution = random.choice(self.distribution_list)
       distance_array = random_distribution.get_array_from_distribution(size=size)
       distance_array = distance_array*maximum_distance
       return distance_array
    
class Errorterm_mixture_distance_distribution_uniform(IErrorterm_mixture_distance_distribution_collection):
    def __init__(self):
        self.distribution_list=[Distribution_uniform(min=0.5, max=0.6)] 
    def get_distance_array(self, size,maximum_distance):
       random_distribution = random.choice(self.distribution_list)
       distance_array = random_distribution.get_array_from_distribution(size=size)
       distance_array = distance_array*maximum_distance
       return distance_array

class Errorterm_mixture_distance_distribution_normal_mixture(IErrorterm_mixture_distance_distribution_collection):
    def __init__(self):
        self.distribution=Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.4], sigma=1)
    def get_distance_array(self, size,maximum_distance):
       distribution = self.distribution
       distance_array = distribution.get_array_from_distribution(size=size)
       distance_array = distance_array*maximum_distance
       return distance_array

class Complex_distribution_list_of_Mus_maker:
    def __init__(self,number_of_maximum_modes: int ,  total_number_of_elements: int, distribution_for_distances_between_components: IErrorterm_mixture_distance_distribution_collection,  maximum_distance_between_modes_factor=5) -> None:
        if number_of_maximum_modes <=0:
            raise ValueError("number of modes must be >0")
        else:
            pass
        self.maximum_distance_between_modes_factor = maximum_distance_between_modes_factor
        self.total_number_of_elements = total_number_of_elements
        self.number_of_maximum_modes = number_of_maximum_modes
        self.distribution_for_distances_between_components = distribution_for_distances_between_components

    def return_sigma(self)-> float:
        sigma = self.maximum_distance/1.5
        return sigma
    def sample_lists_of_mus(self, size_of_samples:int ,  maximum_total_deviation:float)-> List[List[float]]:
        lists_of_mus = []
        modes = random.randint(1, self.number_of_maximum_modes)
        self.maximum_total_deviation = maximum_total_deviation
        self._calculate_maximum_distance_for_mixture_model(modes=modes)
        self._make_list_for_components_per_mode(modes=modes)
        for i in range(size_of_samples):
            modes = random.randint(1, self.number_of_maximum_modes)
            self._make_list_for_components_per_mode(modes=modes)
            single_list_of_mus = self._make_list_of_mus_for_mixture_model(modes=modes)
            lists_of_mus.append(single_list_of_mus)
        return lists_of_mus

    def _calculate_maximum_distance_for_mixture_model(self,modes:int): # max distances between elements, + borders(4)+ distances between modes
        denominator = (self.total_number_of_elements-1)+4+(modes*(self.maximum_distance_between_modes_factor-1))
        maximum_distance = self.maximum_total_deviation/denominator
        self.maximum_distance = maximum_distance

    def _make_list_for_components_per_mode(self, modes:int):
        components_per_mode = [1 for _ in range(modes)]
        # Distribute the remaining elements (if any) after each mode has at least 1
        remaining_elements = self.total_number_of_elements - modes
        for i in range(remaining_elements):
            components_per_mode[i % modes] += 1
        self.components_per_mode = components_per_mode
        print(components_per_mode)

    def _make_list_of_mus_for_mixture_model(self, modes:int ) ->List[float]:

        components_per_mode = self.components_per_mode
        maximum_distance = self.maximum_distance
        maximum_distance_between_modes_factor = self.maximum_distance_between_modes_factor
        distribution = self.distribution_for_distances_between_components
        list_of_mus = [maximum_distance * 2]

        for mode_index in range(modes):
            elements_in_mode = components_per_mode[mode_index]
            if elements_in_mode > 1:
                size_descending_array = random.randint(1, elements_in_mode - 1)
                size_ascending_array = elements_in_mode - 1 - size_descending_array
                distances_array_descending = distribution.get_distance_array(size=size_descending_array, maximum_distance=maximum_distance)
                distances_array_descending = np.sort(distances_array_descending)[::-1]
                distances_array_ascending = distribution.get_distance_array(size=size_ascending_array, maximum_distance=maximum_distance)
                distances_array_ascending = np.sort(distances_array_ascending)

                for i in distances_array_descending:
                    new_mu = list_of_mus[-1] + i
                    list_of_mus.append(new_mu)
                for i in distances_array_ascending:
                    new_mu = list_of_mus[-1] + i
                    list_of_mus.append(new_mu)

            if mode_index < modes - 1:
                distance_between_modes_factor = np.random.uniform(low=1.5, high=maximum_distance_between_modes_factor)
                x = list_of_mus[-1] + (maximum_distance * distance_between_modes_factor)
                list_of_mus.append(x)
        list_of_mus = list_of_mus[:self.total_number_of_elements]
        return list_of_mus
        


class IInterpolation_model(metaclass=ABCMeta):
    @abstractstaticmethod
    def return_number_of_required_values( dimensions:int)->float:
        """Interface Method"""
    @abstractmethod
    def set_interpolator_model(self,values:List[float], dimensions:int):
        """Interface Method"""

    @abstractmethod
    def calculate_interpolated_values(self, inputs): 
        """_summary_

        Args:
            inputs (_type_): _description_
        """

    def plot_interpolator_model(self, label="interpolated component of errorterm"):
        dimensions = self.data_points.shape[1]
        if dimensions == 1:
            self._plot_1d()
        elif dimensions == 2:
            self._plot_2d()
        elif dimensions > 2:
            self._plot_multidim()   
    
    def _plot_1d(self):
        """ For 1D, we use a simple line plot"""
        x = np.linspace(np.min(self.data_points), np.max(self.data_points), 200)
        y = self.calculate_interpolated_values(inputs=x[:, np.newaxis])
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data_points, self.values, color='red', label='Original Data Points')
        plt.plot(x, y, color='blue', label='Interpolated Values')
        plt.xlabel('X-axis')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
    
    def _plot_2d(self):
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
        plt.title("interpolation of one mixture component")
        plt.show()
   
    def _plot_multidim(self):
    
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
        plt.title("interpolation of one mixture component")
        plt.show()



class Interpolation_model_RBFInterpolator_datapoints_grid(IInterpolation_model): # we need controlls for the string used to set the kernel
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


class Interpolation_model_RBFInterpolator_datapoints_random(IInterpolation_model): # we need controlls for the string used to set the kernel
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
  


    interpolator = Interpolation_model_RBFInterpolator_datapoints_grid()
    number = interpolator.return_number_of_required_values(dimensions=2)
    print(number)
    list_interpolation_models = []
    mus_listen_maker = Complex_distribution_list_of_Mus_maker(number_of_maximum_modes=2, total_number_of_elements=10, distribution_for_distances_between_components=Errorterm_mixture_distance_distribution_normal_mixture(),maximum_distance_between_modes_factor=5)
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
        
    x_inputs = [0.75]
    new_mixture = []
    for element in list_interpolation_models:
        value_mu = element.calculate_interpolated_values( inputs= np.array([x_inputs]))
        print(value_mu)
        new_mixture.append(value_mu[0])
    print(new_mixture)
    distribution = Distribution_mixture_of_normals(list_of_mus=new_mixture, sigma=sigma)
    distribution.plot_distribution(label= f"interpolated mixture at input x= {x_inputs}    " )




    #mu = mu[0]
    #sigma= mus_listen_maker.return_sigma()

    #distribution = Distribution_mixture_of_normals(list_of_mus=mu, sigma=sigma)
    #distribution.plot_distribution(label= "bimodal complex -  close modes    " )
                                    
