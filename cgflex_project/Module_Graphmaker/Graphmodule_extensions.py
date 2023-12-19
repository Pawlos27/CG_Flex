""" this module contains classes focusing in processing and analysing already existing graph structures"""



from cgflex_project.Shared_Classes.NodeObject import Nodeobject

import os
import pickle
import dill
from typing import Any, List, Type, Tuple, Optional
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractstaticmethod, abstractmethod

class IGraph_processor(metaclass=ABCMeta):
    """Abstract base Class as INterface: IMplementations of this class contain functionality for analyzing, visualizing, and printing graphs"""
    @abstractmethod
    def load_graph(self, nodelist: List[Nodeobject]):
        """loading the graph into the processor

        Args:
            nodelist (List[Nodeobject]): the nodelist represents the graph structure
        """

    @abstractmethod
    def show_graph(self, plot_title="DAG_Graph"):
        """plotting the graph, can use different libraries for that

        Args:
            plot_title (str, optional): Is giving the plot a title, defaults to "DAG_Graph".
        """

    @abstractmethod
    def make_verticelist(self):
     """creates a list of vertices used for reprecenting the edges"""

    @abstractmethod
    def print_nodelist(self):
     """printing the list of nodes in tabular form"""


class Graph_processor_networx_solo(IGraph_processor):
    """implementation using networkx library to provide the functionalities """
    def __init__(self):
        self.nodelist = None
        self.verticelist = None

    def load_graph(self, nodelist: List[Nodeobject]):
        self.nodelist = nodelist
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph")
        else:
            self.verticelist = self.make_verticelist()
        
    def show_graph(self, plot_title="DAG_Graph"):
        positions_f= self._make_positions_dictionary_top_down()
        self.print_nodelist()
        # Create a color map as a dictionary
        color_map = {node.id: 'red' if node.source else ('yellow' if node.sink else 'grey') for node in self.nodelist}
        G = nx.DiGraph()
        G.add_edges_from(self.verticelist)
        # Assign colors to nodes based on the color_map
        node_colors = [color_map.get(node) for node in G.nodes()]
        if len(positions_f) > 50:
            nodesize = 200
        elif len(positions_f) > 80:
            nodesize = 100
        else: 
            nodesize = 300
        plt.figure(figsize=(12, 10))
        nx.draw_networkx(G, positions_f ,with_labels=True, node_color=node_colors, edge_color='lightblue', node_size=300)
        plt.ylabel("Layer-Axis")
        plt.xlabel("Thorus-Axis  (first dimension)")
        plt.title(plot_title)
        plt.show()

    def show_graph_layered(self, plot_title="DAG_Graph"):
        positions_f= self._make_positions_dictionary_top_down_layer_focus()
        self.print_nodelist()
        # Create a color map as a dictionary
        color_map = {node.id: 'red' if node.source else ('yellow' if node.sink else 'grey') for node in self.nodelist}
        G = nx.DiGraph()
        G.add_edges_from(self.verticelist)
        # Assign colors to nodes based on the color_map
        node_colors = [color_map.get(node) for node in G.nodes()]
        plt.figure(figsize=(12, 10))
        nx.draw_networkx(G, positions_f ,with_labels=True, node_color=node_colors, edge_color='grey', node_size=300)
        plt.ylabel("Layer-Axis")
        plt.xlabel("Thorus-Axis  (first dimension)")
        plt.title(plot_title)
        plt.show()

    def _make_positions_dictionary_top_down_layer_focus(self):
        nodelist = self.nodelist
        positions={}
        nodelist.reverse()
        for node in nodelist:
            positions[node.id]=[node.coord_t[0], -node.layer]
        return positions 

    def _make_positions_dictionary_top_down(self):
        nodelist = self.nodelist
        positions={}
        nodelist.reverse()
        for node in nodelist:
            positions[node.id]=[node.coord_t[0], -node.coord_l]
        return positions
    def _make_positions_dictionary_horizontal(self):
        nodelist = self.nodelist
        positions={}
        nodelist.reverse()
        for node in nodelist:
            positions[node.id]=[node.coord_l,node.coord_t[0]]
        return positions
    
    def make_verticelist(self):
        verticelist=[]
        for node in self.nodelist:
            for j in node.children:
                kante = (node.id, j)
                verticelist.append(kante)
        return verticelist

    def print_nodelist(self):
        df=pd.DataFrame(self.nodelist)
        print (df)


