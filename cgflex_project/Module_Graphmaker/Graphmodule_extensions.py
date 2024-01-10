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
    """Abstract base Class as Interface: Implementations of this class contain functionality for analyzing, visualizing, and printing graphs"""
    @abstractmethod
    def load_graph(self, nodelist: List[Nodeobject]):
        """loading the nodelist into the class

        Args:
            nodelist (List[Nodeobject]): the nodelist represents the graph structure
        """

    @abstractmethod
    def show_graph(self, plot_title="DAG_Graph"):
        """plotting the graph, can use different libraries for that

        Args:
            plot_title (str, optional): Is giving the plot a title, defaults to "DAG_Graph".
        """
    def show_graph_layered(self, plot_title="DAG_Graph"):
        """plotting the graph in a layer perspective

        Args:
            plot_title (str, optional): Is giving the plot a title, defaults to "DAG_Graph".
        """

    @abstractmethod
    def make_edgelist(self):
        """creates a list of vertices used for reprecenting the edges"""

    @abstractmethod
    def print_nodelist(self):
        """printing the list of nodes in tabular form"""

    @abstractmethod
    def return_graph_metrics(self):
        """returning metrics of graph"""
    
    @abstractmethod
    def print_graph_metrics(self):
        """returning metrics of graph"""

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
            self.verticelist = self.make_edgelist()
        
    def show_graph(self, plot_title="DAG_Graph"):
        positions_f= self._make_positions_dictionary_top_down()
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


    def print_graph_metrics(self):
        x = self.return_graph_metrics
        print (x)

    def return_graph_metrics(self):
        G = nx.DiGraph()
        G.add_edges_from(self.verticelist)
        metrics = self._calculate_metrics(G=G)
        df=pd.DataFrame(metrics)
        return df

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

    @staticmethod
    def _calculate_metrics(G):
  
        sources = [node for node, deg in G.in_degree() if deg == 0]
        sinks = [node for node, deg in G.out_degree() if deg == 0]
        metrics = {
            "number_of_nodes": [G.number_of_nodes()],
            "number_of_edges": [G.number_of_edges()],
            #"average_shortest_path_length": [nx.average_shortest_path_length(G)],
            "density": [nx.density(G)],
            "longest_path_length": [len(nx.dag_longest_path(G))],
            "max_degree": [max(G.degree(), key=lambda item: item[1])[1]],
            "average_in_degree": [sum(dict(G.in_degree()).values()) / G.number_of_nodes()],
            "average_out_degree": [sum(dict(G.out_degree()).values()) / G.number_of_nodes()],
            "average_degree": [sum(dict(G.degree()).values()) / G.number_of_nodes()],
            "max_closeness": [max(nx.closeness_centrality(G).items(), key=lambda x: x[1])[1]],
            "max_betweenness": [max(nx.betweenness_centrality(G).items(), key=lambda x: x[1])[1]],
            "number_of_sources": [len(sources)],
            "number_of_sinks": [len(sinks)]}

        return metrics
        


    def _make_positions_dictionary_top_down_layer_focus(self):
        nodelist = self.nodelist
        positions={}
        nodelist.reverse()
        for node in nodelist:
            positions[node.id]=[node.coord_t[0], -node.layer]
        nodelist.sort()
        return positions 

    def _make_positions_dictionary_top_down(self):
        nodelist = self.nodelist
        positions={}
        nodelist.reverse()
        for node in nodelist:
            positions[node.id]=[node.coord_t[0], -node.coord_l]
        nodelist.sort()
        return positions
    def _make_positions_dictionary_horizontal(self):
        nodelist = self.nodelist
        positions={}
        nodelist.reverse()
        for node in nodelist:
            positions[node.id]=[node.coord_l,node.coord_t[0]]
        nodelist.sort()
        return positions
    
    def make_edgelist(self):
        verticelist=[]
        for node in self.nodelist:
            for j in node.children:
                kante = (node.id, j)
                verticelist.append(kante)
        return verticelist

    def print_nodelist(self):
        df=pd.DataFrame(self.nodelist)
        print (df)


