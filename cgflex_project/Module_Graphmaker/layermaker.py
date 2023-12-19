"""
This module  contains classes that take care of processing a list of NodeObjects. Specifically, they are only concerned with managing the 'Layer' attribute, setting the Layer Attribute to assign a certain Layer to the node."

.. note::
   This module requires external dependencies such as the 'Nodeobject' Class.
"""

from abc import ABCMeta, abstractmethod
import math
from typing import  List, Type
from cgflex_project.Shared_Classes.NodeObject import Nodeobject






class ILayersetter(metaclass=ABCMeta):
    """
    Abstract base class and Interface. 
    Implemented classes should implement the make_layer method to define specific logic for assigning layers to nodes.
    """
    @abstractmethod
    def make_layer(self, nodelist: List[Nodeobject])-> List[Nodeobject]:
        """ method for setting layers in a nodelist."""


class Layer_setter_equinumber(ILayersetter):
    """
    Class for assigning layers to Nodeobjects so that each layer contains an approximately equal number of nodes.

    Args:
        layernumber (int): The number of layers to create.
    """
    def __init__(self, layernumber: int):
        self.layernumber = layernumber
    def make_layer(self, nodelist: list):
          
        """Assigns layers to each node in the list based on an equal distribution of nodes per layer"""
        nodelist.sort()

        nodelist_f = nodelist
        layernumber_f=self.layernumber
        nodes_per_layer= math.floor(len(nodelist_f)/layernumber_f)
     
        counter = 0
        for node in nodelist_f:
            node.layer = (math.floor(counter/nodes_per_layer))
            counter += 1
        return nodelist_f


class Layer_setter_equispace(ILayersetter):
        
    """Class for assigning layers to Nodeobjects. Here the Layers are dicided across the total space of the Layerdimension, and it is simply checked in which Layer the koordinates are.

    
    Args:
        layernumber (int): The number of layers to create.
    """
    
    def __init__(self,layernumber: int):
        self.layernumber = layernumber
    def make_layer( self, nodelist: List[Nodeobject]):
        """Assigns layers to each node in the list dependending of the position of the node, based on an equal spacing of the Layers across the total space"""
        nodelist.sort()

        nodelist_f = nodelist
        layernumber_f = self.layernumber
        scalerange=nodelist_f[1].scale_val
        scalesteps= scalerange/layernumber_f

        for j in range(layernumber_f):
            min = j*scalesteps
            max = (j+1)*scalesteps

            for node in nodelist_f:
                coordinatelayer = node.coord_l 
                if (coordinatelayer >= min) and (coordinatelayer < max ) :
                    node.layer = j
        return nodelist_f

class Layer_setter_equispace_last_first_solo(ILayersetter):
    """Class for assigning layers to Nodeobjects based on equal spacing, with the first and last nodes being in separate layers

    Args:
        layernumber (int): The number of layers to create, not including layers for the first and last nodes.
    """


    def __init__(self, layernumber: int):
        self.layernumber = layernumber
    def make_layer( self, nodelist: List[Nodeobject]):
        """Assigns layers to each node in the list based on their position in a scaled range. The first and last nodes 
        are assigned to separate layers."""
        nodelist.sort()

        nodelist_f = nodelist
        layernumber_f=self.layernumber
        scalerange=nodelist_f[1].scale_val
        scalesteps= scalerange/layernumber_f

        for j in range(layernumber_f):
            min = j*scalesteps
            max = (j+1)*scalesteps

            for node in nodelist_f:
                coordinatelayer = node.coord_l 
                if (coordinatelayer >= min) and (coordinatelayer < max ) :
                    node.layer = j+1
        nodelist_f[0].layer=0
        nodelist_f[-1].layer= layernumber_f+1
        return nodelist_f



class Layer_setter_continuous(ILayersetter):
    """Class for assigning a unique layer to each Nodeobject in the list, essentially numbering them sequentially.
    This class does not use any attributes and parameters."""
    def __init__(self):
        pass
    def make_layer(self,nodelist: List[Nodeobject]):
        """Assigns a unique layer to each node in the list, effectively numbering them in order."""
        nodelist.sort()

        nodelist_f = nodelist
        counter = 0
        for node in nodelist_f:
            node.layer = counter
            counter += 1
        return nodelist_f



