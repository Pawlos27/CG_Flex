from abc import ABCMeta, abstractstaticmethod, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Type
from cgflex_project.Shared_Classes.distributions import *

class IInitial_value_distribution_collection(metaclass=ABCMeta):
    @abstractmethod
    def get_distribution(self)-> IDistributions:
     """Interface Method"""


class Initial_value_distribution_random_full(IInitial_value_distribution_collection):
    def __init__(self):
        self.distribution_list=[Distribution_uniform(min=0, max=1),Distribution_normal_truncated_at_3sigma(mu=0.5,sigma=0.16666),Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.0, 0.075,  0.15,  0.225,   0.3,   0.375,  0.625,  0.7,   0.775,  0.85, 0.925,], sigma=0.05)] 
    def get_distribution(self):
       random_distribution = random.choice(self.distribution_list)
       return random_distribution
    

class Initial_value_distribuiton_edged(IInitial_value_distribution_collection):
    def __init__(self):
        self.distribution=Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.0, 0.075,  0.15,  0.225,   0.3,    0.7,   0.775,  0.85, 0.925,], sigma=0.05)
    def get_distribution(self)-> IDistributions:
       distribution = self.distribution
       return distribution

class Initial_value_distribution_uniform(IInitial_value_distribution_collection):
    def __init__(self):
        self.distribution= Distribution_uniform(min=0, max=1)
    def get_distribution(self)-> IDistributions:
       distribution = self.distribution
       return distribution
    
class Initial_value_distribution_normal_truncated(IInitial_value_distribution_collection):
    def __init__(self):
        self.distribution= Distribution_normal_truncated_at_3sigma(mu=0.5,sigma=0.16666)
    def get_distribution(self)-> IDistributions:
       distribution = self.distribution
       return distribution
