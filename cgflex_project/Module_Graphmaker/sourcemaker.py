"""
This module  contains classes that take care of processing a list of NodeObjects. Specifically, they are only concerned with managing the 'Source' attribute, setting nodes to sources."

.. note::
   This module requires external dependencies such as the 'Nodeobject' Class.
"""
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import random
from typing import  List, Type
from cgflex_project.Shared_Classes.NodeObject import Nodeobject


def correct_sources_and_sinks(nodelist: List[Nodeobject])-> List[Nodeobject]:
        """
        This function corrects the source and sink attributes of the nodes in the nodelist, it makes sure these two attributes are always consistent.
        First node always source, last node never source, last node always sink.

        Args:
            nodelist (List[Nodeobject]): The list of Nodeobjects to be corrected.

        Returns:
            List[Nodeobject]: The updated list of Nodeobjects with corrected source and sink attributes.
        """

        nodelist[0].source = True
        nodelist[0].sink= False
        nodelist[-1].source = False
        nodelist[-1].sink = True
        return nodelist

class ISource_setter(metaclass=ABCMeta): 
    """
    Interface class for setting sources in a list of Nodeobjects.
    ...
    """
    @abstractmethod
    def make_sources(self, nodelist: List[Nodeobject])-> List[Nodeobject]:
     """Interface method for setting sources in the node list."""
    @abstractstaticmethod
    def _correct_sources(nodelist: List[Nodeobject])-> List[Nodeobject]:
     """Interface Method for making sure, correction is implemented for correcting sources and sinks in the node list
        ..."""


class Source_setter_by_probability(ISource_setter):
    """
    Class for setting sources in a list of Nodeobjects based on a probability calculation, every node is checked in order of the index (layer downwards), until enough sources are found. The first node is always source.
  
    Args:
        number_of_sources (int): The number of sources to be set in the node list.
        shift_parameter (int, optional): A parameter to adjust the probability of a node being a source. Defaults to 2, a higher value will cause sources to appear earlier in the search.
    """
    def __init__(self,number_of_sources: int, shift_parameter=2):


        self.number_of_sources = number_of_sources
        self.shift_parameter =  shift_parameter
    def make_sources(self,nodelist: List[Nodeobject]):
        nodelist.sort()

        number_of_sources = self.number_of_sources
        nodelist_f = nodelist
        number_of_nodes = len(nodelist_f)
        for node in nodelist_f :    # reset all sources
            node.source = False
        count = 1
        while count < number_of_sources:    # set new sources
            for j in range(1,number_of_nodes-1):
                if nodelist_f[j].source == False and nodelist_f[j].sink == False  :
                    nodelist_f[j].source = self._sources_probabilitycalc(number_of_nodes=number_of_nodes)
                    if nodelist_f[j].source == True :
                        count += 1
                    if count == number_of_sources:
                        break
        nodelist = self._correct_sources(nodelist=nodelist_f) # correct
        return nodelist
    def _sources_probabilitycalc(self, number_of_nodes): 
        """function givin out true/false values,  multiplicator is applied to shift the occurance to the upper part of the graph"""
        multiplicator = self.shift_parameter                               
        probability = self.number_of_sources*multiplicator/number_of_nodes  # probability of the appearence is calculated by expectedprobability for the number of required sources
        return random.random() < probability
    @staticmethod
    def _correct_sources(nodelist: List[Nodeobject])-> List[Nodeobject]:
        nodelist = correct_sources_and_sinks(nodelist=nodelist)
        return nodelist



""" Sources made by handmade list , first Node always source"""
class Source_setter_handmade(ISource_setter):
    """
    A class for setting source nodes based on a manually specified list of node IDs.
    implements the method for setting sources as per a provided list of node IDs.

    Args:
        list_of_sources (list): A list of integers representing the IDs of nodes to be set as sources.
    """


    def __init__(self, list_of_sources:List[int]):

        self.list_of_sources= list_of_sources


    def make_sources(self,nodelist: List[Nodeobject]):

        """Sets specified nodes as sources based on the list_of_sources attribute."""
        nodelist.sort()
        for node in nodelist:     # reset all sources
            node.source = False
        for source in self.list_of_sources: # set new sources
            nodelist[source].source = True
        nodelist = self._correct_sources(nodelist=nodelist) # correct
        return nodelist
    @staticmethod
    def _correct_sources(nodelist: List[Nodeobject])-> List[Nodeobject]:
        nodelist = correct_sources_and_sinks(nodelist=nodelist)
        return nodelist




