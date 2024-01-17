""""""
"""
This module contains all classes related to the generation of edges in the graph. 
It includes implementations of the IEdgemaker class, which performs node generation, as well as implementations 
of the IEdge_probability_by_distance class, which implement a weighting function that defines how the distance
 between nodes affects the probability of an edge

"""

from abc import ABCMeta,abstractmethod
import random
from typing import  List, Type
from cgflex_project.Shared_Classes.NodeObject import Nodeobject
import math





class IEdge_probability_by_distance(metaclass=ABCMeta):
    """ 
    This interface defines weighting functions that determine how the distance between two nodes influences the formation of an edge. The crucial aspect in these functions is the slope of the curve they generate. 
    This slope indicates how rapidly the probability of edge formation changes with the distance between nodes. The steeper the slope, the more significant the impact of distance on edge formation. 

    Important note: It's important to mention that weighting functions can be static or dynamic. A static function remains constant over time, whereas a dynamic function adapts to the expanding space. This means that the characteristics of the weighting are dependent on the relative distance in the entire space. This is based on the principle that the space in which the nodes are located grows as the number of nodes increases.

    The use of a dynamic function would avoid the situation where a graph with more nodes grows organically; instead, it becomes denser in a specific manner.

    It is also important to note that separate functions can be used for layer distance and torus distance. 

    """
    @abstractmethod
    def probability_calc(self, distance:float, scale_factor:float)-> float:
        """ Calculates the weight regarding to the distance 

        Args:
            distance (float): the distance between 2 nodes
            scale_factor (float): the scale factor is a regulator for the steepness in the weight functions



        Returns:
            float: The weight regarding to the distance.
        """


class Edge_probability_by_distance_decreasing_power_law_like(IEdge_probability_by_distance):

    def __init__(self,  type_static_dynamic = "static",exponential_factor=2.0):
        self.exponential_factor = exponential_factor
        self.type = type_static_dynamic
        if self.exponential_factor <=0:
            raise ValueError("factor in probability per distance calculation needs do be >0")
    def probability_calc(self, distance:float, scale_factor:float):
        if self.type == "static":
            scaling= 1
        elif self.type == "dynamic":
            scaling = scale_factor
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        probability = (1/(1+(distance/scaling))**(self.exponential_factor))
        return probability
    
class Edge_probability_by_distance_decreasing_exponentially(IEdge_probability_by_distance):

    def __init__(self,  type_static_dynamic = "static", exponential_factor= 1.0):
        self.exponential_factor = exponential_factor
        self.type = type_static_dynamic
        if self.exponential_factor <=0:
            raise ValueError("factor in probability per distance calculation needs do be >0")
    def probability_calc(self, distance:float, scale_factor:float):
        if self.type == "static":
            scaling= 1
        elif self.type == "dynamic":
            scaling = scale_factor
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        probability = math.e **(-distance*self.exponential_factor)
        return probability

class Edge_probability_by_distance_decreasing_inverse(IEdge_probability_by_distance):

    def __init__(self,  type_static_dynamic = "static", offset= 0,exponential_factor=1.0 ):
        self.offset = offset
        self.exponential_factor = exponential_factor
        self.type = type_static_dynamic
        if self.exponential_factor <=0:
            raise ValueError("factor in probability per distance calculation needs do be >0")
        if self.offset <0:
            raise ValueError("offset factor in probability per distance calculation needs do be >0")
    def probability_calc(self, distance:float, scale_factor:float):
        if self.type == "static":
            scaling= 1
        elif self.type == "dynamic":
            scaling = scale_factor
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        probability = 1/(1+(distance/scaling*self.exponential_factor)+self.offset)
        return probability

class Edge_probability_by_distance_decreasing_linear(IEdge_probability_by_distance):

    def __init__(self, type_static_dynamic = "dynamic", manual_scaling=1):
        self.type = type_static_dynamic
        self.manual_scaling = manual_scaling
        if self.manual_scaling <0:
            raise ValueError("manual scaling  factor in probability per distance calculation needs do be >0")
    def probability_calc(self, distance:float, scale_factor:float):
        if self.type == "static":
            scaling= 1
        elif self.type == "dynamic":
            scaling = scale_factor
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        probability = 1- (distance/scaling/self.manual_scaling)
        if probability == 0:
            probability = 0
        return probability

class Edge_probability_by_distance_uniform(IEdge_probability_by_distance):

    def __init__(self, probability=1):
        self.probability = probability
    def probability_calc(self, distance:float, scale_factor:float):
        probability = 1
        return probability

class Edge_probability_by_distance_decreasing_linear(IEdge_probability_by_distance):

    def __init__(self, type_static_dynamic = "dynamic", manual_scaling=1):
        self.type = type_static_dynamic
        self.manual_scaling = manual_scaling
        if self.manual_scaling <0:
            raise ValueError("offset  factor in probability per distance calculation needs do be >0")
    def probability_calc(self, distance:float, scale_factor:float):
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        distance_f = distance
        probability =  1 - math.log(1 + distance * self.manual_scaling)
        if probability > 1:
            probability = 1
        if probability < 0:
            probability = 0.1
        
        return probability 






class IEdgemaker(metaclass=ABCMeta):
    """ Abstract method to create edges in the node list. An interface whose implementations define a method for creating edges in a Nodelist. Initially edges are generated by evaluating each possible pair of nodes, taking into account the topological order (a node can only connect with nodes below it). A set of constraints is examined for this purpose; if the pair meets these constraints, an edge is generated. Then, edges are established by adding a list of IDs of connected nodes to the child or parent attributes of a NodeObject — in the child attribute for directly subordinate nodes and in the parent attribute for preceding nodes.

    Args:
        nodelist (List[Nodeobject]): The list of nodes to establish connections between.

    Returns:
        List[Nodeobject]: The updated list of nodes with established connections."""

    @abstractmethod
    def make_edges(self, nodelist: List[Nodeobject])-> List[Nodeobject]:
        """The abstract method make_edges is responsible in the implementations for creating edges in the node list. It should implement a logic that checks potential node pairs to determine if there is a connection between them. If a connection exists, these should be entered into the Nodelist
        """


class edgemaker(IEdgemaker):
    """ A class that creates edges between nodes in the Nodelist based on specified criteria and probabilities. In a top-down approach, all possible connections are evaluated for their probability.In this implementation, no connections are possible within the same layer, and the probability of edge formation is dependent on weighting functions over the distance of the nodes. Separate functions can be chosen for the torus and the layer distance.The probability is scaled with a c-factor, which is determined beforehand to match the overall sum of probabilities with the expected amount of edges we want.

    Args:
        layerlimit (int): The limit for layers edges can cross.
        number_of_edges (float): The number of edges to be established.
        edgeprobability_layer (IEdge_probability_by_distance): Weight function for impact of layer-distance on edge generation.
        edgeprobability_thorus (IEdge_probability_by_distance): Weight function for impact of thorus-distance on edge generation.
        edge_outdegree_max (int): The maximum out-degree of an edge.
        edge_indegree_max (int): The maximum in-degree of an edge.
        inner_layer_connection (bool): Flag to allow connections within the same layer.


    """
   
    def __init__(self,layerlimit:int,number_of_edges:float, edge_outdegree_max: int , edge_indegree_max: int ,edgeprobability_thorus: IEdge_probability_by_distance = Edge_probability_by_distance_decreasing_exponentially(),edgeprobability_layer: IEdge_probability_by_distance = Edge_probability_by_distance_decreasing_exponentially(),  inner_layer_connection=False ):
        self.layerlimit = layerlimit
        self.number_of_edges = int(number_of_edges*0.9)
        self.edgeprobability_layer = edgeprobability_layer
        self.edgeprobability_thorus = edgeprobability_thorus
        self.edge_outdegree_max = edge_outdegree_max
        self.edge_indegree_max = edge_indegree_max
        self.inner_layer_connection = inner_layer_connection

    def make_edges(self,nodelist: List[Nodeobject])-> List[Nodeobject]:
        """ Creates edges in the provided nodelist based on defined weight functions and constraints. This method first calculates the scaling factor c, then it iteratively checks pairs of nodes regarding the provided constraints and calculates the probability of edges based on the function and the c_Factor. In the first loop, the edges are generated as the basic structure. In the second loop, missing incoming and outgoing edges are added to the nodes.
        
        Args:
            nodelist (List[Nodeobject]): The list of nodes to establish connections between.

        Returns:
            List[Nodeobject]: The updated list of nodes with established connections.
        """
        self.nodelist = nodelist
        self.nodelist.sort()
        self._initialize_parameters()
        self._reset_edges()
        self._first_loop_edgemaking()
        self._second_loop_edgemaking_correcting_outgoing_edges()
        self._third_loop_edgemaking_correcting_ingoing_edges()
        self.sort_parents_children(nodelist=nodelist)
        return nodelist
    
    def sort_parents_children(self,nodelist: List[Nodeobject])-> List[Nodeobject]:
        for node in nodelist:
            node.parents.sort()
            node.children.sort()

    def  _initialize_parameters(self):
        self.listlenght = len(self.nodelist)
        self.dimensions_thorus = len(self.nodelist[0].coord_t)
        self.scale_factor = self.nodelist[0].scale_val
        self.maximum_distance_thorus = self._calculate_maximum_thorus_distance(scale_factor=self.scale_factor, dimensions=self.dimensions_thorus)
        self._parameter_c_calc()

    def _reset_edges(self):
            for node in self.nodelist:
                node.parents = []
                node.children = []

    def _first_loop_edgemaking(self):
        """ The first loop in the edge making process. This loop iteratively checks and establishes connections between nodes based on the defined rules and probabilities. It focuses on creating a basic structure of edges before further refinement in subsequent loops.
        """
        
        for n, node in enumerate(self.nodelist):
            for j in range(n+1,self.listlenght ):
                if node.sink == False and self.nodelist[j].source == False and len(node.children) < self.edge_outdegree_max and len(self.nodelist[j].parents) < self.edge_indegree_max and self.nodelist[j].layer - node.layer < self.layerlimit: 
                    if self.inner_layer_connection == False and self.nodelist[n].layer == self.nodelist[j].layer:
                        if self.nodelist[n].source == True or self.nodelist[j].sink == True:
                            self._check_and_make_edge(parent_id=n,child_id=j)
                        else: 
                            pass
                    else:
                        self._check_and_make_edge(parent_id=n,child_id=j)
            
    def _second_loop_edgemaking_correcting_outgoing_edges(self):
        """
        The second loop for edge making, focusing on correcting outgoing edges.

        This loop ensures that nodes designated as non-sink nodes have at least one outgoing edge. It iterates through the 
        nodes, attempting to establish edges where none exist, adhering to the specified constraints and probabilities.

        """
        
        for n, node in enumerate(self.nodelist):
            if node.sink == False and len(node.children) == 0 :
                counter_1 = 0
                while len(self.nodelist[n].children) < 1 and counter_1<3000:
                    for j in range(n+1,self.listlenght ):
                        if self.nodelist[n].sink == False and self.nodelist[j].source == False and len(self.nodelist[n].children) < self.edge_outdegree_max and len(self.nodelist[j].parents) < self.edge_indegree_max and self.nodelist[j].layer - self.nodelist[n].layer < self.layerlimit:
                            if self.inner_layer_connection == False and self.nodelist[n].layer == self.nodelist[j].layer:
                                if self.nodelist[n].source == True or self.nodelist[j].sink == True:
                                    make_edge = self._check_and_make_edge(parent_id=n,child_id=j,solo_edge_condition=True)
                                    if make_edge == True :
                                        break
                            else:
                                make_edge = self._check_and_make_edge(parent_id=n,child_id=j,solo_edge_condition=True)
                                if make_edge == True :
                                    break
                    counter_1 += 1 
                #we have to set sink if it didnt find any partner after 50 tries
                if len(self.nodelist[n].children) == 0:
                    node.sink = True
        
    def _third_loop_edgemaking_correcting_ingoing_edges(self): #modularization required
        """
        The third loop for edge making, focusing on correcting ingoing edges.

        This loop ensures that nodes designated as non-source nodes have at least one ingoing edge. It iterates in reverse 
        through the nodes, establishing necessary edges to meet the ingoing edge requirements, based on the defined 
        constraints and probabilities.

        """
        self.nodelist.reverse()
        for n, node in enumerate(self.nodelist):
            if self.nodelist[n].source == False and len(self.nodelist[n].parents) == 0 :
                counter = 0 
                while len(self.nodelist[n].parents) == 0 and counter<3000:
                    for j in range(n+1,self.listlenght ):
                        if self.nodelist[j].sink == False and  self.nodelist[n].layer - self.nodelist[j].layer  < self.layerlimit and len(self.nodelist[j].children) < self.edge_outdegree_max and len(self.nodelist[n].parents) < self.edge_indegree_max :
                            if self.inner_layer_connection == False and self.nodelist[n].layer == self.nodelist[j].layer:
                                if self.nodelist[j].source == True or self.nodelist[n].sink == True:
                                    make_edge = self._check_and_make_edge(parent_id=j,child_id=n, solo_edge_condition=True)
                                    if make_edge == True :
                                        #self.nodelist[j].children.append(self.nodelist[n].id)
                                        #self.nodelist[n].parents.append(self.nodelist[j].id)
                                        break
                           
                            make_edge = self._check_and_make_edge(parent_id=j,child_id=n, solo_edge_condition=True)
                            if make_edge == True :
                                #self.nodelist[j].children.append(self.nodelist[n].id)
                                #self.nodelist[n].parents.append(self.nodelist[j].id)
                                break
                    counter += 1 
                if len(self.nodelist[n].parents) == 0:
                    node.source = True               
        self.nodelist.reverse()

    def _check_probability_for_possible_edge(self, parent_id:int, child_id:int):
        """
        Checks the probability of an edge existing between two nodes.

        This method calculates the probability of an edge existing between two nodes, identified by their IDs, based on 
        the distance calculations for layer and thorus coordinates.

        Args:
            parent_id (int): The ID of the potential parent node.
            child_id (int): The ID of the potential child node.

        Returns:
            float: The calculated probability of an edge existing between the two nodes.
        """
        n=parent_id
        j=child_id
        t_distance_f = self._edge_distance_calc_thorus(coord_t_parent=self.nodelist[n].coord_t, coord_t_child=self.nodelist[j].coord_t )
        l_distance= self._edge_distance_calc_layer(coord_l_parent=self.nodelist[n].coord_l, coord_l_child=self.nodelist[j].coord_l)
        edge_probability = self._edge_probability_calc(t_distance=t_distance_f, l_distance=l_distance)
        return edge_probability

    def _check_and_make_edge(self, parent_id:int, child_id:int, solo_edge_condition = False):
        """
        Checks and establishes an edge between two nodes if conditions are met. This method checks if an edge should be established between two nodes based on the calculated probability. 
        
        Args:
            parent_id (int): The ID of the parent node.
            child_id (int): The ID of the child node.
            solo_edge_condition (bool, optional): A flag to indicate if the edge is being created under special conditions. 
            Defaults to False.

        Returns:
            bool:
        """
        edge_probability = self._check_probability_for_possible_edge(parent_id=parent_id, child_id=child_id)
        edgecheck = self._edgechecker(edge_probability)
        if edgecheck == True :
            self.nodelist[parent_id].children.append(self.nodelist[child_id].id)
            self.nodelist[child_id].parents.append(self.nodelist[parent_id].id)
            if solo_edge_condition == True:
                return True

    def _parameter_c_calc(self) ->float:
        """
        Calculates the parameter 'c' used in edge probability calculations.

        This method determines the value of 'c' to ensure the expected number of edges aligns with the target number of edges, based on the current configuration and probabilities.
        It follows roughly the same methodology as the edge calculation as it is aiming to correct its results by a scaling factor.

        Returns:
            float: The calculated value of the parameter 'c'.
        """
        nodelist = self.nodelist
        number_of_edges_f=self.number_of_edges
        nodelist= nodelist
        self.c= 1
        lower_c_limit=0
        upper_c_limit= None
        expected_edges = 0
        while expected_edges  < 0.8*number_of_edges_f or expected_edges > 1.2*number_of_edges_f :
            n = 0
            expected_edges = 0
            for node in nodelist:
                for j in range(n+1,self.listlenght) :
                    if nodelist[n].sink == False and nodelist[j].source == False and nodelist[n].layer != nodelist[j].layer and nodelist[n].layer != nodelist[j].layer:
                        if self.inner_layer_connection == False and self.nodelist[n].layer == self.nodelist[j].layer:
                            if self.nodelist[n].source == True or self.nodelist[j].sink == True:
                                self._check_and_make_edge(parent_id=n,child_id=j)
                            else: 
                                pass
                        else:
                            edge_probability = self._check_probability_for_possible_edge(parent_id=n,child_id=j)
                            expected_edges += edge_probability
                n+=1
            if expected_edges  < 0.95*number_of_edges_f:
                lower_c_limit= self.c
                if upper_c_limit ==  None:
                    self.c= 2*self.c
                else:
                    self.c= (upper_c_limit+lower_c_limit)/2

            elif expected_edges > 1.05*number_of_edges_f:
                upper_c_limit= self.c
                if lower_c_limit == 0:
                    self.c= self.c/2
                else:
                    self.c= (upper_c_limit+lower_c_limit)/2
        
   
    @staticmethod
    def _edge_distance_calc_layer(coord_l_parent:float, coord_l_child:float)->float:
        """
        Calculates the distance between two points in an 1d -carthesian space for layer coordinates

        :param coord_l_parent: coordinate for the first point.
        :param coord_l_child: coordinate for the second point.
        :return: The distance between the two points in the layer dimension
        """
        l_distance = coord_l_child - coord_l_parent
        return l_distance

    def _edge_distance_calc_thorus(self,coord_t_parent:list, coord_t_child:list)-> float:
        """
        Calculates the distance between two points in an n-dimensional torus.

        :param coord_t_parent: List of coordinates for the first point.
        :param coord_t_child: List of coordinates for the second point.
        :param torus_length: The length of the torus in each dimension, identical with the scaling value, es the standard lenght is 1.
        :return: The distance between the two points in the torus.
        """
        # same lenght?
        if len(coord_t_parent) != len(coord_t_child):
            return "List of torus coordinates have to be of same lenght."
        squared_sum = 0
        # sistance per dimension
        for i in range(len(coord_t_parent)):
            dim_distance = min(abs(coord_t_parent[i] - coord_t_child[i]), self.scale_factor - abs(coord_t_parent[i] - coord_t_child[i]))
            # adding square to total sum
            squared_sum += dim_distance ** 2
        distance = squared_sum ** 0.5
        distance = abs(distance)
        return distance

    def _edge_probability_calc(self,t_distance:float, l_distance:float )->float:
        """ calculates the probability of the occurance of an edge, given the distances, the probability_by_distance_ functions and the Correction faktor we call c factor

        Args:
            t_distance (float): distance between torus coordinates
            l_distance (float): distance between layer coordinates
            edgeprobability_thorus (IEdge_probability_by_distance): weight function for torus distance
            edgeprobability_layer (IEdge_probability_by_distance): weight function for layer distance
            c_faktor (float): scaling factor C

        Returns:
            float: the actual probability of edge generation
        
        """
        edgeprobability_thorus = self.edgeprobability_thorus
        edgeprobability_layer = self.edgeprobability_layer
        edge_prob_t= edgeprobability_thorus.probability_calc(distance=t_distance, scale_factor=self.maximum_distance_thorus)
        edge_prob_l= edgeprobability_layer.probability_calc(distance=l_distance, scale_factor= 1)
        edge_probability_f=  (self.c*edge_prob_l*edge_prob_t)**2
        if edge_probability_f > 1:
                edge_probability_f  = 1
        return edge_probability_f

    def _calculate_maximum_thorus_distance(self,dimensions:int, scale_factor:float)-> float:
        """ helper function with formula for calc. maximum distance in thorus

        Args:
            dimensions (int): dimensions of the thorus
            scale_factor (float): scale factor is the same as the lenght of one side of the thorus

        Returns:
            float: the maximum value the thorus can achieve 
        """

        maximum_distance = (scale_factor / 2) * (dimensions ** 0.5)
        return maximum_distance

    @staticmethod
    def _edgechecker(probability)-> bool :
        return random.random() < probability
              

