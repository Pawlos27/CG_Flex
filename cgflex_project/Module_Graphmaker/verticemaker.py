#prüfen ob kein sink, source, ob anzahl an elternknoten/kinderknoten nicht überschritten 

# thorus distanz hinzufügen, scalator beachten
# verschiedene
#
# c ist korrekturwert um erwarteten knotengrad zu erreichen , c^2 verwenden? [ für c müsste auch nicht im selben layer regel gelten]

# ingoin edges loop evtl fehler bei j,n / children parents < max knotengrade, breaks an richtiger stelle?

from abc import ABCMeta,abstractmethod
import random
from typing import  List, Type
from cgflex_project.Shared_Classes.NodeObject import Nodeobject
import math





class IEdge_probability_by_distance(metaclass=ABCMeta):
    """ This Bastract Base Class is the INetrface for classes which calculate the probability of occurence of edges, related to a distance measure,
    so given a distance measure they return a probability.

    Noteworthy is that the space is scaled, some of the probabily calculators are static , some are dynamic and adapt to the scaled space, but it doesnt mean that they take the same stretch factor and recycle it, in spacec, the maximum achievable distance 
    """
    @abstractmethod
    def probability_calc(self, distance:float, scale_factor:float):
        """_summary_

        Args:
            distance (float): _description_
            scale_factor (float): the scale factor is an important argument, it is 

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
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
    """  Diese implementation stellt eine formel da in der eine sub_expotentielle abnahme stattfindet

    also durch 
    """
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
    @abstractmethod
    def make_edges(self, nodelist: List[Nodeobject])-> List[Nodeobject]:
     """Interface Method"""

class edgemaker(IEdgemaker): # bei 2ter runde suchen evtl warscheinlichkeit multiplizieren wenn andere knoten schon kinder haben
    def __init__(self,layerlimit:int,number_of_edges:float,edgeprobability_layer: IEdge_probability_by_distance, edgeprobability_thorus: IEdge_probability_by_distance, edge_outdegree_max: int , edge_indegree_max: int , inner_layer_connection=False ):
        self.layerlimit = layerlimit
        self.number_of_edges = number_of_edges
        self.edgeprobability_layer = edgeprobability_layer
        self.edgeprobability_thorus = edgeprobability_thorus
        self.edge_outdegree_max = edge_outdegree_max
        self.edge_indegree_max = edge_indegree_max
        self.inner_layer_connection = inner_layer_connection

    def make_edges(self,nodelist: List[Nodeobject])-> List[Nodeobject]:
        self.nodelist = nodelist
        self.nodelist.sort()
        self._initialize_parameters()
        self._reset_edges()
        self._first_loop_edgemaking()
        self._second_loop_edgemaking_correcting_outgoing_edges()
        self._third_loop_edgemaking_correcting_ingoing_edges()
        return nodelist
    
    def  _initialize_parameters(self):
        self.listlenght = len(self.nodelist)
        self.dimensions_thorus = len(self.nodelist[0].coord_t)
        self.scale_factor = self.nodelist[0].scale_val
        self.maximum_distance_thorus = self._calculate_maximum_thorus_distance(scale_factor=self.scale_factor, dimensions=self.dimensions_thorus)
        self._parameter_c_calc()
        print (f"final c is {self.c}" )
        print(self.c)

    def _reset_edges(self):
            for node in self.nodelist:
                node.parents = []
                node.children = []

    def _first_loop_edgemaking(self):
        
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
        The loop checks if all nodes who are no sinks have an incoming edge, if not it adds "one"
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
                    print(f"Verticcemaker: the second correction loop round:   {counter}")
                if len(self.nodelist[n].parents) == 0:
                    node.source = True               
        self.nodelist.reverse()

    def _check_probability_for_possible_edge(self, parent_id:int, child_id:int):
        n=parent_id
        j=child_id
        t_distance_f = self._edge_distance_calc_thorus(coord_t_parent=self.nodelist[n].coord_t, coord_t_child=self.nodelist[j].coord_t )
        l_distance= self._edge_distance_calc_layer(coord_l_parent=self.nodelist[n].coord_l, coord_l_child=self.nodelist[j].coord_l)
        edge_probability = self._edge_probability_calc(t_distance=t_distance_f, l_distance=l_distance)
        return edge_probability

    def _check_and_make_edge(self, parent_id:int, child_id:int, solo_edge_condition = False):
        edge_probability = self._check_probability_for_possible_edge(parent_id=parent_id, child_id=child_id)
        edgecheck = self._edgechecker(edge_probability)
        if edgecheck == True :
            print("edge found in first round")
            self.nodelist[parent_id].children.append(self.nodelist[child_id].id)
            self.nodelist[child_id].parents.append(self.nodelist[parent_id].id)
            if solo_edge_condition == True:
                return True

    def _parameter_c_calc(self) ->float:
        """zuerst wird der expected value abhängig von den distanzen und der angewanten verteilung berechnet"""
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
            return "Die Listen müssen dieselbe Länge haben."
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
            t_distance (float): _description_
            l_distance (float): _description_
            edgeprobability_thorus (IEdge_probability_by_distance): _description_
            edgeprobability_layer (IEdge_probability_by_distance): _description_
            c_faktor (float): correction factor C

        Returns:
            _type_: _description_
        
        """
        edgeprobability_thorus = self.edgeprobability_thorus
        edgeprobability_layer = self.edgeprobability_layer
        edge_prob_t= edgeprobability_thorus.probability_calc(distance=t_distance, scale_factor=self.maximum_distance_thorus)
        edge_prob_l= edgeprobability_layer.probability_calc(distance=l_distance, scale_factor= 1)
        print(f"c ist {self.c}, edge prob ist {edge_prob_l} und {edge_prob_t}")
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
              

