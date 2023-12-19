"""this Module contains the controll class for Graphs """
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






class Graph_controller:
    """_summary_

    Args:
        config (Blueprint_graph): _description_
    """
    def __init__(self, config: Blueprint_graph ):
        """_summary_

        Attributes:
            config (Blueprint_graph): holds the config object, which holds all the necessary sub classs instances the controller works with
            nodelist(): contains the nodelist after it is created, updated after every manipulation on graph level
        """
        self.nodelist = None
        self.config = config


    def make_graph(self):
        """this method uses the functionality of the class objects contained in the config object to create the graph
        """
        nodelist_raw = self.config.nodemaker.make_nodelist()
        nodelist_layered = self.config.layermaker.make_layer(nodelist= nodelist_raw)
        nodelist_sourced = self.config.sourcemaker.make_sources(nodelist= nodelist_layered )
        nodelist_sinked = self.config.sinkmaker.make_sinks(nodelist= nodelist_sourced)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sinked)

    def reset_config(self, config: Blueprint_graph):
        """replaces the config object for a new one

        Args:
            config (Blueprint_graph): new config
        """
        self.config= config

    def reset_layers(self):
        """resets the layers of the graph, after replacing config , otherwise layers would stay the same"""
        nodelist_layered = self.config.layermaker.make_layer(nodelist=self.nodelist)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_layered)

    def new_sources(self):
        """resets sources and generates new, implies also new edges
        """
        nodelist_sourced = self.config.sourcemaker.make_sources(nodelist=self.nodelist)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sourced)
       
    def new_sinks(self ):
        nodelist_sinked = self.config.sinkmaker.make_sinks(nodelist=self.nodelist)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sinked)

    def new_sinks_and_sources(self):
        nodelist_sourced = self.config.sourcemaker.make_sources(nodelist=self.nodelist)
        nodelist_sinked = self.config.sourcemaker.make_sources(nodelist=nodelist_sourced)
        self.nodelist = self.config.edgemaker.make_edges(nodelist=nodelist_sinked)
     
    def new__edges(self):
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
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        self.config.graphprocessor.show_graph()

    def showgraph_layer_perspective(self, plot_title="DAG_Graph_by layer"):
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        self.config.graphprocessor.show_graph()
        

    def print_nodelist(self):
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        self.config.graphprocessor.print_nodelist()



    def get_edgelist(self):
        self.config.graphprocessor.load_graph(nodelist=self.nodelist)
        edgelist = self.config.graphprocessor.make_edgelist()
        return edgelist
    
        
    def get_list_of_sources(self):
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph")
        else:
            list_of_sources = []
            for node in self.nodelist:
                if node.source == True:
                    list_of_sources.append(node.id)
            return list_of_sources

    def get_list_of_sinks(self):
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph")
        else:
            list_of_sinks = []
            for node in self.nodelist:
                if node.sink== True:
                    list_of_sinks.append(node.id)
            return list_of_sinks

    def get_nodelist_graph(self)-> List[Type[Nodeobject]]: 
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph")
        else:
            return self.nodelist





    


if __name__ == "__main__":
    from cgflex_project.Shared_Classes.config_objects import blueprint_test
    newgraph= Graph_controller(config=blueprint_test)

    initilaization = True
    if initilaization == True:
        newgraph.make_graph()
        newgraph.print_nodelist()
        newgraph.showgraph()
        newgraph.safe_nodelist_pikle(file_name="pikle1",file_path=None)
    elif initilaization == False:
        newgraph.load_nodelist_pikle(file_name="pikle1", file_path=None)
        newgraph.showgraph()

