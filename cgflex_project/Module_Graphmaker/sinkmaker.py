"""
This module  contains classes that take care of processing a list of NodeObjects. Specifically, they are only concerned with managing the 'Sink' attribute, setting nodes to sources."

.. note::
   This module requires external dependencies such as the 'Nodeobject' Class.
"""
from cgflex_project.Module_Graphmaker.sourcemaker import correct_sources_and_sinks
from typing import  List, Type
from cgflex_project.Shared_Classes.NodeObject import Nodeobject
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import random




class ISink_setter(metaclass=ABCMeta):
    """
    Interface class for setting sinks in a list of Nodeobjects.
    ...
    """
    @abstractmethod
    def make_sinks(self, nodelist: List[Nodeobject])-> List[Nodeobject]:
     """ method for setting sources in the node list."""
    @abstractstaticmethod
    def _correct_sinks(nodelist: List[Nodeobject])-> List[Nodeobject]:
     """Method for making sure, correction is implemented for correcting sources and sinks in the node list
        ..."""


class Sink_setter_by_probability(ISink_setter):
    """
    Class for setting sinks in a list of Nodeobjects based on a probability calculation, every sink is checked in order of the index (layer upwards), until enough sinks are found. The last node is always sink.
  
    Args:
        number_of_sinks (int): The number of sinks to be set in the node list.
        shift_parameter (int, optional): A parameter to adjust the probability of a node being a sink. Defaults to 2, a higher value will cause sinks to appear lower in the structure.
    """
    def __init__(self, number_of_sinks: int, shift_parameter=2):
        self.number_of_sinks =  number_of_sinks
        self.shift_parameter =  shift_parameter
    def make_sinks(self,nodelist: List[Nodeobject]):
        nodelist.sort()

        number_of_sinks = self.number_of_sinks
        nodelist_f = nodelist
        nodelist_f.reverse()
        number_of_nodes = len(nodelist_f)
        for node in nodelist_f :    # reset all sinks
            node.sink = False
        nodelist_f[0].sink = True    # set new sinks
        count = 1
        while count < number_of_sinks:
            for j in range(1,number_of_nodes-1):
                if nodelist_f[j].sink == False and nodelist_f[j].source == False :
                    nodelist_f[j].sink = self._sinks_probabilitycalc(number_of_nodes=number_of_nodes)
                    if nodelist_f[j].sink == True :
                        count += 1
                    if count == number_of_sinks:
                        break
        nodelist_f.reverse()
        nodelist = self._correct_sinks(nodelist=nodelist_f)
        return nodelist
    
    def _sinks_probabilitycalc(self,number_of_nodes=int):  
        """function givin out true/false values,  multiplicator is applied to shift the occurance of sinks to the lower part of the graph"""

        multiplicator = self.shift_parameter       
        probability = self.number_of_sinks*(multiplicator/number_of_nodes)
        return random.random() < probability
    @staticmethod
    def _correct_sinks(nodelist: List[Nodeobject]):
        nodelist = correct_sources_and_sinks(nodelist=nodelist)
        return nodelist



class Sink_setter_handmade(ISink_setter):
    """
    A class for setting sink nodes based on a manually specified list of node IDs.
    implements the method for setting sinks as per a provided list of node IDs.

    Args:
        list_of_sinks (list): A list of integers representing the IDs of nodes to be set as sinks.
    """
    def __init__(self,list_of_sinks : List[int]):
        self.list_of_sinks = list_of_sinks 
    def make_sinks(self,nodelist : List[Nodeobject] ):
        
        """Sets specified nodes as sinks based on the list_of_sinks attribute."""
        nodelist.sort()
        for node in nodelist: # reset sinks
            node.sink = False
        for sink in self.list_of_sinks:
            if nodelist[sink].source == False:
                nodelist[sink].sink = True
        nodelist = self._correct_sinks(nodelist=nodelist)
        return nodelist

    @staticmethod
    def _correct_sinks(nodelist: List[Nodeobject]):
        nodelist = correct_sources_and_sinks(nodelist=nodelist)
        return nodelist
    
        

if __name__ == "__main__":
    pass