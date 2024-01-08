from abc import ABCMeta, abstractstaticmethod, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Type, Tuple
from cgflex_project.Shared_Classes.distributions import *

class IInitial_value_distribution_collection(metaclass=ABCMeta):

    @abstractmethod
    def initialize_distributions(self, range: Tuple[float,float]):
     """Interface Method"""
    @abstractmethod
    def get_distribution(self)-> IDistributions:
     """Interface Method"""


class Initial_value_distribution_random_full(IInitial_value_distribution_collection):
    def __init__(self):
        pass

    def initialize_distributions(self, range: Tuple[float,float]):
        self.distribution_list = [Distribution_mixture_of_normals_truncated_at_3sigma_random_sigma_outward_random_inside_borders_and_uniform_at_construction(lower_border= range[0], upper_border= range[1], components=5),
                                  Distribution_uniform(min= range[0],max= range[1]),
                                  Distribution_normal(mu= ( range[1] +  range[0])/2,sigma= (abs( range[1] -  range[0]))/6),
                                  Distribution_uniform_random_at_construction(min_absolute= range[0], max_absolute=  range[1]),
                                  Distribution_normal_truncated_at_3sigma_random_all_inside_borders_random_at_construction(lower_border= range[0], upper_border= range[1]),
                                  Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(lower_border= range[0], upper_border= range[1])]
    def get_distribution(self):
       random_distribution = random.choice(self.distribution_list)
       return random_distribution
    
class Initial_value_distribution_random_mixture(IInitial_value_distribution_collection):
    def __init__(self):
        pass

    def initialize_distributions(self, range: Tuple[float,float]):
        self.distribution_list = [Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(lower_border= range[0], upper_border= range[1])]
    def get_distribution(self):
       random_distribution = random.choice(self.distribution_list)
       return random_distribution
    


if __name__ == "__main__":
    range = ( 0, 4)
    #distrib =  Initial_value_distribution_random_full() 
    distrib = Initial_value_distribution_random_mixture() 
    distrib.initialize_distributions(range=range)
    distribi = distrib.get_distribution()
    distribi.plot_distribution(label="initial distribution")

    pass