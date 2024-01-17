"""
This module is the first of a couple of modules providing classes and functionality for managing node objects. It creates an initial list of Nodeobjects and focuses on defining various implementable collections on distributions, and the Nodemaker, which generates a list of Nodes using these distributions. Nodes are distributed over space, and the module includes abstract base classes and their implementations for the distributions as well as the Nodemaker.

.. note::
   This module requires external dependencies such as the 'distributions' module, as the distribution_collections instantiate distribution classes from this module.
"""



from dataclasses import dataclass, field
from itertools import count
from cgflex_project.Shared_Classes.distributions import *
from typing import Any, List, Type, Union
from abc import ABCMeta, abstractstaticmethod, abstractmethod
from cgflex_project.Shared_Classes.NodeObject import Nodeobject

class INodemaker_distribution_collection(metaclass=ABCMeta):
    """
    Abstract base class to represent a collection of distributions used in nodemaker.
    This class serves as an interface for different nodemaker distribution collections, 
    ensuring they implement the required method to get a distribution.
    """

    @abstractmethod
    def get_distribution(self)-> IDistributions:
        """
        :return: an instance of a distribution class 
        :rtype: IDistributions """



class Nodemaker_distribution_random_full(INodemaker_distribution_collection):

    """ Implements a random full distribution for nodemaker.

    This class provides a way to randomly select a distribution from the predefined full list of  instaciated distributions. """
    
    def __init__(self):
        """ Initializes the distribution list with specific distribution instances. """
        self.distribution_list=[Distribution_uniform(min=0, max=1),
                                Distribution_normal_truncated_at_3sigma(mu=0.5,sigma=0.16666),
                                Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.0, 0.075,  0.15,  0.225,   0.3,   0.375,  0.625,  0.7,   0.775,  0.85, 0.925,], sigma=0.05),
                                Distribution_mixture_of_normals_truncated_at_3sigma_outward_random_inside_borders_and_uniform_at_construction(sigma= 0.07, components= 9),
                                Distribution_mixture_of_normals_truncated_at_3sigma_inwards_random_uniform_at_construction(sigma= 0.08, components= 11),
                                Distribution_mixture_of_normals_controlled_modes_complex_spread_of_mus_and_random(lower_border= 0, upper_border= 1)] 
    def get_distribution(self):
        """ Randomly selects and returns a distribution from the distribution list.
        """
        random_distribution = random.choice(self.distribution_list)
        return random_distribution
    

class Nodemaker_distribuiton_bottleneck(INodemaker_distribution_collection):
    """
    Implements a bottleneck distribution for nodemaker.
    This class provides a specific customized distribution of  mixture of truncated_normals, which together form a distribution with a single bottleneck
    """
    def __init__(self):
        self.distribution=Distribution_mixture_of_normals_truncated_custom(list_of_mus=[0.0, 0.075,  0.15,  0.225,   0.3,   0.375,   0.7,   0.775,  0.85, 0.925,], sigma=0.05)
    def get_distribution(self)-> IDistributions:
       distribution = self.distribution
       return distribution

class Nodemaker_distribution_uniform(INodemaker_distribution_collection):
    """
    Implements and returns a single uniform distribution"""
    def __init__(self):
        self.distribution= Distribution_uniform(min=0, max=1)
    def get_distribution(self)-> IDistributions:
       distribution = self.distribution
       return distribution
    
class Nodemaker_distribution_normal_truncated(INodemaker_distribution_collection):
    """
    Implements and returns a single normal distribution"""
    def __init__(self):
        self.distribution= Distribution_normal_truncated_at_3sigma(mu=0.5,sigma=0.16666)
    def get_distribution(self)-> IDistributions:
       distribution = self.distribution
       return distribution




class INodemaker(metaclass=ABCMeta):
    """
    Abstract base class for creating nodelists.
    This class serves as an interface for different nodemaker implementations, 
    ensuring they implement the required method to make a nodelist.
    """
    @abstractmethod
    def make_nodelist(self)-> List[Nodeobject]:
        """Creates and returns a list of Nodeobject instances
        
        """

class Nodemaker(INodemaker):
    """
    Implementation of INodemaker for creating a list of node objects.
    Provides functionality to create a list of node objects based on 
    given distributions for both layer and thorus dimensions.
    
    Args:
        number_of_nodes(int) : The number of nodes to create.
        number_of_dimensions_thorus (int):  The number of dimensions for thorus coordinates.
        l_distribution (INodemaker_distribution_collection): The distribution collection used for layer coordinates.
        t_distribution (INodemaker_distribution_collection): The distribution collection used for thorus coordinates.
        scale_per_n (float): The scale factor applied to the nodes' space, dependent on the number of nodes.
 

    """
    def __init__(self, number_of_nodes: int,number_of_dimensions_thorus: int = 1 , l_distribution : INodemaker_distribution_collection = Nodemaker_distribution_uniform(), t_distribution: INodemaker_distribution_collection = Nodemaker_distribution_uniform(),scale_per_n:float=0.05) :

        
        self.number_of_nodes = number_of_nodes

        self.dimensions = number_of_dimensions_thorus

        self.l_distribution = l_distribution 

        self.t_distribution = t_distribution

        self.scale_value = scale_per_n * number_of_nodes
 

    def make_nodelist(self) -> List[Nodeobject]:
        """ Creates and returns a list of Nodeobject instances based on the coordinates retrieved from  predefined distributions.
        This method utilizes the provided layer and thorus distributions to generate a list
        of nodes, each with its own unique ccoordinates.

        :return: A list of Nodeobject instances, each representing a node in the network.
        :rtype: List[Type[Nodeobject]] """

        coordinate_array_layer = self._layer_coordinates_generation()
        coordinate_list_thorus = self._thorus_coordinates_generation()
        nodelist = []
        for j in range(self.number_of_nodes):
            node = Nodeobject(id=j, coord_t=coordinate_list_thorus[j], coord_l=coordinate_array_layer[j], scale_val=self.scale_value)
            nodelist.append(node)
            nodelist.sort()
        nodelist = self._set_id_in_nodelist(nodelist=nodelist)
        return nodelist
    
    @staticmethod
    def _set_id_in_nodelist(nodelist: List[Type[Nodeobject]])-> List[Type[Nodeobject]] :
        """the id is set so it maches with the list index which is alligned with the layer_coordinates. """
        counter = 0
        for node in nodelist:
            node.id = int(counter)
            counter +=1
        return nodelist
    
    def _layer_coordinates_generation(self):
        distribution_layer = self.l_distribution.get_distribution()
        coordinate_array_layer= distribution_layer.get_array_from_distribution(size=self.number_of_nodes)
        coordinate_array_layer = coordinate_array_layer * self.scale_value
        return coordinate_array_layer
    def _thorus_coordinates_generation(self):
        scale_value = self.scale_value
        number_of_nodes=self.number_of_nodes
        dimensions=self.dimensions
        distribution=self.t_distribution.get_distribution()
        x = distribution.get_array_from_distribution(size=number_of_nodes*dimensions)
        x = x * scale_value 
        t_coord_list = []
        counter = 0
        for j in range(number_of_nodes):
            y=[]
            for n in range(dimensions):
                z= x[counter]
                y.append(z)
                counter += 1
            t_coord_list.append(y)
        return t_coord_list











if __name__ == "__main__":

    #a = nodelistmaker(number_of_nodes=10, number_of_dimensions_thorus=3, scale_per_n=0.1, l_distribution=Distribution_uniform(), t_distribution=Distribution_uniform())
    #b=a[1].scale_val

    nodelist= Nodemaker(number_of_dimensions_thorus=2,number_of_nodes=50,scale_per_n=0.1,l_distribution=Nodemaker_distribution_uniform(),t_distribution=Nodemaker_distribution_uniform()).make_nodelist()

    print(nodelist)
