""" This module contains the main controller class which provides and coordinates all functions for graph generation, manipulation, and presentation. """
from cgflex_project.Shared_Classes.blueprints import Blueprint_graph
from cgflex_project.Shared_Classes.NodeObject import Nodeobject
import pandas as pd
import os
import pickle
import dill
from typing import Any, List, Type, Tuple, Optional
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cgflex_project.Module_Graphmaker.sinkmaker import Sink_setter_handmade
from cgflex_project.Module_Graphmaker.sourcemaker import Source_setter_handmade







class Graph_controller:
    """  The controller coordinates all subclasses associated with graph creation, manipulation, or analysis, and represents an interface for the graph generation functionality of the framework. The methods are reused by the main controller, but the controller and its methods can also be used independently by users


    Args:
        config (Blueprint_graph): A configuration object that provides instances of the classes used by the controller
    """
    def __init__(self, config: Blueprint_graph ):
        """
        Attributes:
            config (Blueprint_graph): holds the config object, which holds all the necessary sub classs instances the controller works with
            nodelist(List[Nodeobject]): contains the nodelist after it is created, updated after every manipulation on graph level
        """
        self.nodelist = None
        self.config = config


    def make_graph(self):
        """Uses the functionality of class objects contained in the config object to create the graph. It sequentially calls methods from different components of the config object to create 
        and modify the nodelist, forming a graph
         """
        nodelist_raw = self.config.nodemaker.make_nodelist()
        nodelist_layered = self.config.layermaker.make_layer(nodelist= nodelist_raw)
        nodelist_sourced = self.config.sourcemaker.make_sources(nodelist= nodelist_layered )
        nodelist_sinked = self.config.sinkmaker.make_sinks(nodelist= nodelist_sourced)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sinked)

    def reset_config(self, config: Blueprint_graph):
        """ Replaces the existing config object with a new one.

        Args:
            config (Blueprint_graph): The new configuration object for the graph.
        """
        self.config= config

    def reset_layers(self):
        """
        Resets the layers of the graph. This method is used after replacing the config object to ensure the layers are set accordingly with the new configuration.
        """
        nodelist_layered = self.config.layermaker.make_layer(nodelist=self.nodelist)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_layered)

    def new_sources(self, list_of_sources:List[int]= None ):
        """
        Resets sources and generates new ones, implying also the creation of new edges.
        """
        if list_of_sources == None :
            nodelist_sourced = self.config.sourcemaker.make_sources(nodelist=self.nodelist)
            self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sourced)
        else:
            if max(list_of_sources) >= len(self.nodelist):
                raise ValueError("given source ids are not in the nodelist, too big")
            source_maker = Source_setter_handmade(list_of_sources=list_of_sources)
            nodelist_sourced = source_maker.make_sources(nodelist=self.nodelist)
            self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sourced)
       
    def new_sinks(self,list_of_sinks:List[int]= None ):
        """
        Resets sinks in the graph and generates new ones,implying also the creation of new edges.
        """
        if list_of_sinks == None :
            nodelist_sinked = self.config.sinkmaker.make_sinks(nodelist=self.nodelist)
            self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sinked)

        else:
            if max(list_of_sinks) >= len(self.nodelist):
                raise ValueError("given sink ids are not in the nodelist, too big")
            sink_maker = Sink_setter_handmade(list_of_sinks=list_of_sinks)
            nodelist_sinked = sink_maker.make_sinks(nodelist=self.nodelist)
            self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sinked)


    def new_sinks_and_sources(self):
        """
        Resets both sinks and sources in the graph and generates new ones and new edges.
        """
        nodelist_sourced = self.config.sourcemaker.make_sources(nodelist=self.nodelist)
        nodelist_sinked = self.config.sourcemaker.make_sources(nodelist=nodelist_sourced)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sinked)
     
    def new__edges(self):
        """Generates new edges for the existing nodelist in the graph.
        """
        self.nodelist = self.config.edgemaker.make_edges(nodelist=self.nodelist)

    def safe_nodelist_pikle(self,file_name:str, file_path:Optional[str]):
        if self.nodelist is not None:
            if file_path == None:
                data_folder = os.path.join(os.path.dirname(__file__), 'data')
                file_path = os.path.join(data_folder, file_name)

            with open(file_path, 'wb') as file:
                pickle.dump(self.nodelist, file)
        else:
            raise ValueError("No data to save. Please set data using set_data method first.")
        
    def load_nodelist_pikle(self,file_name:str,file_path:Optional[str]):
        if file_path == None:
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'rb') as file:
            self.nodelist = pickle.load(file)


    def showgraph(self, plot_title="DAG_Graph"):
        """
        Displays the graph as a diagram using the graph processor class.

        Args:
            plot_title (str): The title for the graph plot.
        """
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        self.config.graphprocessor.show_graph(plot_title=plot_title)



    def showgraph_layer_perspective(self, plot_title="DAG_Graph_by layer"):
        """
        Displays the graph sorting the nodes according to its layers, more orderly representation.

        Args:
            plot_title (str): The title for the graph plot.
        """

        
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        self.config.graphprocessor.show_graph_layered()
        
    def print_graph_metrics(self, plot_title="DAG_Graph"):
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        self.config.graphprocessor.print_graph_metrics()

    def return_graph_metrics(self):
        """
        Returns the metrics of the graph, provided by networx library, stored ina dataframe.

        Returns:
            Various metrics calculated for the graph in a pandas dataframe.
        """
        
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        x = self.config.graphprocessor.return_graph_metrics()
        return x

    def print_nodelist(self):
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        self.config.graphprocessor.print_nodelist()



    def get_verticelist(self):
        """
        Generates and returns a list of vertices in the graph, creating a list of node pairs.

        Returns:
            List of vertices in the graph.
        """
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        edgelist = self.config.graphprocessor.make_edgelist()
        return edgelist
    
        
    def get_list_of_sources(self):
        """
        Retrieves a list of source nodes in the graph.

        Returns:
            List of IDs of source nodes.

        Raises:
            ValueError: If there is no graph yet.
        """
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph")
        else:
            list_of_sources = []
            for node in self.nodelist:
                if node.source == True:
                    list_of_sources.append(node.id)
            return list_of_sources

    def get_list_of_sinks(self):
        """
        Retrieves a list of sink nodes in the graph.

        Returns:
            List of IDs of sink nodes.

        Raises:
            ValueError: If there is no graph yet.
        """
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph")
        else:
            list_of_sinks = []
            for node in self.nodelist:
                if node.sink== True:
                    list_of_sinks.append(node.id)
            return list_of_sinks

    def get_nodelist_graph(self)-> List[Type[Nodeobject]]: 
        """
        Returns the nodelist of the graph.

        Returns:
            List[Type[Nodeobject]]: The current nodelist of the graph.

        Raises:
            ValueError: If there is no graph yet.
        """
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph")
        else:
            return self.nodelist





    
