
from typing import Any, List, Type, Tuple, Optional, Union
import random
import copy
from cgflex_project.Main_Controller.Controllermodule_extensions import IObject_serializer, IController_coordinator
from  cgflex_project.Module_Dependencymaker.Dependencymodule import Dependency_controller
from cgflex_project.Module_Sampler.Samplingmodule import Sampling_controller
from cgflex_project.Module_Graphmaker.Graphmodule import Graph_controller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cgflex_project.Shared_Classes.blueprints import Blueprint_graph, Blueprint_dependency, Blueprint_sampling
from cgflex_project.Shared_Classes.blueprint_main_controller import Blueprint_main_controller

 

class Cg_flex_controller:
    """
    Class Cg_flex_controller manages the coordination of graph and dependency controllers,
    along with handling sampling processes, this is the main interface for the user, the methods lets the user acess
    the functionality mostly provided by the subcontrollers. Except the Serialization which is managed by the controller itself.

    Attributes:
        graph_controller: Manages graph-related operations.
        dependency_controller: Manages dependency-related operations.
        sampling_controller: Manages sampling processes.
        config: Configuration settings for the controller.
        sampling_controller_sync: Flag to indicate if the sampling controller is synchronized.
        dependency_controller_sync: Flag to indicate if the dependency controller is synchronized.
    """
    def __init__(self,config= Blueprint_main_controller):
        """
        Initializes the Cg_flex_controller with the specified configuration.

        Args:
            config (Blueprint_main_controller, optional): Configuration settings for the controller.
        """
        
        self.graph_controller = None
        self.dependency_controller = None
        self.sampling_controller = None
        self.config = config
        self.sampling_controller_sync = False
        self.dependency_controller_sync = False


# methods related to  graph controller 
    def reset_controller(self, size):
        """
        Resets the controller with a specified number of graph and dependency controllers.

        Args:
            size (int): The number of graph and dependency controllers to be reset.
        """
        list_of_graph_controller = self.config.controller_coordinator.make_list_graph_controller(size=size)
        self.graph_controller = list_of_graph_controller
        list_of_dependency_controller = self.config.controller_coordinator.make_list_dependency_controller(size=size)
        self.dependency_controller = list_of_dependency_controller
        self.sampling_controller = self.config.sampling_controller

    def get_number_of_graph_components(self):
        """
        Retrieves the number of graph components managed by the controller.

        Returns:
            int: The number of graph components.
        """
        components = len(self.graph_controller)
        return components
    
    def make_graphs(self, number_of_decoupled_elements:int):
        """
        Creates graphs based on the specified number of decoupled elements.

        Args:
            number_of_decoupled_elements (int): The number of decoupled elements in the graph.
        """
        self.dependency_controller_sync = False
        self.reset_controller(size=number_of_decoupled_elements) 
        for graph_controller in self.graph_controller:
            graph_controller.make_graph()

    def reset_configs_graph(self, config: Blueprint_graph, graph_id:int,):
        """
        Resets the configuration of a specific graph controller by its ID.

        Args:
            config (Blueprint_graph): The new configuration for the graph.
            graph_id (int): The ID of the graph controller to be reset.
        """
        self.graph_controller[graph_id].reset_config(config=config)

    def reset_layers_graph(self, graph_id=0):
        """
        Resets the layers of a specific graph .

        Args:
            graph_id (int, optional): The ID of the graph controller whose layers are to be reset. Defaults to 0.
        """
        self.dependency_controller_sync = False
        self.graph_controller[graph_id].reset_layers()

    def set_new_sources_graph(self,list_of_sources:List[int]= None , graph_id=0):
        """
        Updates the source nodes for a specific graph, the user can simply regenerate or set by hand .

        Args:
            list_of_sources (List[int], optional): A list of new source node IDs. Defaults to None.
            graph_id (int, optional): The ID of the graph controller to update. Defaults to 0.
        """
        self.dependency_controller_sync = False
        self.graph_controller[graph_id].new_sources(list_of_sources=list_of_sources)

    def set_new_sinks_graph(self, list_of_sinks:List[int]= None ,graph_id=0):
        """
        Updates the sink nodes for a specific graph , the user can simply regenerate or set by hand.

        Args:
            list_of_sinks (List[int], optional): A list of new sink node IDs. Defaults to None.
            graph_id (int, optional): The ID of the graph controller to update. Defaults to 0.
        """
        self.dependency_controller_sync = False
        self.graph_controller[graph_id].new_sinks(list_of_sinks=list_of_sinks)

    def set_new_sinks_and_sources_graph(self, graph_id=0):
        """
        resets both the source and sink nodes for a specific graph.

        Args:
            graph_id (int, optional): The ID of the graph controller to update. Defaults to 0.
        """
        self.dependency_controller_sync = False
        self.graph_controller[graph_id].new_sinks_and_sources()

    def set_new_edges_graph(self, graph_id=0):
        """
        resets the edges for a specific graph .

        Args:
            graph_id (int, optional): The ID of the graph controller to update. Defaults to 0.
        """
        self.dependency_controller_sync = False
        self.graph_controller[graph_id].new__edges()

    def print_nodelists_graph(self,  graph_ids: Optional[List[int]] = None):
        """
        Prints the node lists of specified graphs.

        Args:
            graph_ids (Optional[List[int]], optional): A list of graph controller IDs. If None, all graphs are printed. Defaults to None.
        """
        if graph_ids == None:
            for graph_controller in self.graph_controller:
                graph_controller.print_nodelist()
        else:
            for graph_id in graph_ids:
                self.graph_controller[graph_id].print_nodelist()
    
    def print_graph_metrics(self):
        """
        Prints metrics for all graph controllers managed by this controller.
        """ 
        df = self.return_graph_metrics()
        print(df)


    def return_graph_metrics(self):
        """
        Retrieves and returns metrics for all graph controllers.

        Returns:
            DataFrame: A pandas DataFrame containing metrics for all graph controllers.
        """
        x = []
        for graph_controller in self.graph_controller:
            x_2 = graph_controller.return_graph_metrics()
            x.append(x_2)
        df = pd.concat(x, ignore_index=True)
        return df
    

    def get_edgelist_graph(self): 
        """
        Retrieves a nested list of edges for all graph controllers.

        Returns:
            List[List[Any]]: A nested list where each sublist contains vertices of a graph controller.
        """
        nested_verticelist = []
        for graph_controller in self.graph_controller:
            verticelist = graph_controller.get_edgelist()
            nested_verticelist.append(verticelist)
        return nested_verticelist

    def get_sourcelist_graph(self):
        """
        Retrieves a nested list of source nodes for all graphs.

        Returns:
            List[List[int]]: A nested list where each sublist contains source nodes of a graph controller.
        """
        nested_sourcelist = []
        for graph_controller in self.graph_controller:
            sourcelist = graph_controller.get_list_of_sources()
            nested_sourcelist.append(sourcelist)
        return nested_sourcelist
    
    def get_sinklist_graph(self): 
        """
        Retrieves a nested list of sink nodes for all graph controllers.

        Returns:
            List[List[int]]: A nested list where each sublist contains sink nodes of a graph controller.
        """
        nested_sinklist = []
        for graph_controller in self.graph_controller:
            sinklist = graph_controller.get_list_of_sinks()
            nested_sinklist.append(sinklist)
        return nested_sinklist
    
    def get_nodelists_graph(self): # returns nested nodelists of al lgraphs
        """
        Retrieves a nested list of node lists for all graph controllers.

        Returns:
            List[List[NodeObject]]: A nested list where each sublist contains nodes of a graph controller.
        """
        nested_nodelists = []
        for graph_controller in self.graph_controller:
            nodelist = graph_controller.get_nodelist_graph()
            nested_nodelists.append(nodelist)
        return nested_nodelists

    def plot_graph(self,graph_ids: Optional[List[int]] = None): 
        """
        Plots all or specific graphs managed by the controller.

        Args:
            graph_ids (Optional[List[int]], optional): A list of graph controller IDs. If None, all graphs are plotted. Defaults to None.
        """
        if graph_ids == None:
            counter = 0
            for graph in self.graph_controller:
                graph.showgraph(plot_title= f"DAG_Graph Nr:{counter}")
                counter += 1
        else:
            for graph_id in graph_ids:
                    graph_title = f"DAG_Graph Nr:{graph_id}"
                    self.graph_controller[graph_id].showgraph(plot_title=graph_title)
                    

    def plot_graph_by_layer(self,graph_ids: Optional[List[int]] = None): 
        """
        Plots all or specific graphs by layers, as managed by the controller.

        Args:
            graph_ids (Optional[List[int]], optional): A list of graph controller IDs. If None, all graphs are plotted. Defaults to None.
        """
        if graph_ids == None:
            counter = 0
            for graph in self.graph_controller:
                graph.showgraph_layer_perspective(plot_title= f"DAG_Graph by Layer Nr:{counter}")
                counter += 1
        else:
            for graph_id in graph_ids:
                    graph_title = f"DAG_Graph by Layer Nr:{graph_id}"
                    self.graph_controller[graph_id].showgraph_layer_perspective(plot_title=graph_title)
        



# Methods dependencycontroller 
    def load_graph_nodelists_into_dependency_controller(self): 
        """
        Loads nodelists from the graph controllers into the corresponding dependency controllers.
        Adjusts the length of dependency controllers and updates their nodelists.
        """
        self.dependency_controller_sync = True
        for i in range (len(self.graph_controller)):
            self.dependency_controller[i].load_nodelist_graph(nodelist=self.graph_controller[i].get_nodelist_graph())

    def make_dependencies(self):
        """
        Initializes and creates dependencies for all nodes in all dependency controllers.
        Also sets the  sync of the sampling controller to false.
        """
        self.sampling_controller_sync = False
        self.load_graph_nodelists_into_dependency_controller()
        for dependency_controller in self.dependency_controller:
            dependency_controller.make_dependencies()

    def reset_dependencies(self):
        """
        Resets and recreates dependencies for all dependency controllers.
        """
        self.sampling_controller_sync = False
        for dependency_controller in self.dependency_controller:
            dependency_controller.make_dependencies()

    def reset_configs_dependency(self, config: Blueprint_dependency, graph_id=0):
        """
        Resets the configuration of a specified dependency controller.

        Args:
            config (Blueprint_dependency): The new configuration for the dependency controller.
            graph_id (int, optional): The ID of the dependency controller to update. Defaults to 0.
        """
        self.dependency_controller[graph_id].reset_config(config=config)

    def reset_dependencies_specific(self, node_ids: List[int], graph_id=0): 
        """
        Resets dependencies for specific nodes in a specified dependency controller.

        Args:
            node_ids (List[int]): The IDs of the nodes to reset dependencies for.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.sampling_controller_sync = False
        self.dependency_controller[graph_id].reset_dependencies_specific(node_id=node_ids)
    
    def reset_tsd_counter(self):
        """
        Resets the time series data counter in the sampling controller.
        """
        self.sampling_controller.reset_tsd_counter()
    
    def replace_dependencies_by_tsd_function(self, node_ids: List[int], graph_id=0): 
        """
        Replaces dependencies of specified nodes with time series data functions in a specified dependency controller.

        Args:
            node_ids (List[int]): The IDs of the nodes to update dependencies for.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.sampling_controller_sync = False
        self.dependency_controller[graph_id].replace_dependencies_by_tsd_function(node_id=node_ids)

    def replace_dependencies_by_tsd_function_abstract_id(self, ids_abstract:List[int]):
        """
        Replaces dependencies for nodes with abstract IDs using time series data functions.

        Args:
            ids_abstract (List[int]): The abstract IDs of the nodes to update dependencies for.
        """
        self.sampling_controller_sync = False
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.dependency_controller[graph_id].replace_dependencies_by_tsd_function(node_id=[node_id])

    def replace_dependencies_by_initial_distributions(self, node_ids: List[int], graph_id=0):
        """
        Replaces dependencies of specified nodes with initial distributions in a specific dependency controller, making them random values.

        Args:
            node_ids (List[int]): List of node IDs to have their dependencies replaced.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.sampling_controller_sync = False
        self.dependency_controller[graph_id].replace_dependencies_by_initial_distributions(node_ids=node_ids)

    def replace_all_dependencies_by_initial_distributions(self):
        """
        Replaces dependencies of all nodes in each dependency controller with initial distributions, then every nodes value is random.
        """
        self.sampling_controller_sync = False
        for dependency_controller in self.dependency_controller:
            dependency_controller.replace_all_dependencies_by_initial_distributions()

    def replace_dependencies_by_single_values(self, node_ids_and_values: List[Tuple[int, float]] ,graph_id=0):
        """
        Replaces dependencies of specified nodes with single fixed values in a specific dependency controller.

        Args:
            node_ids_and_values (List[Tuple[int, float]]): List of tuples containing node IDs and their corresponding values.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.sampling_controller_sync = False
        self.dependency_controller[graph_id].replace_dependencies_by_single_values(node_ids_and_values=node_ids_and_values)

    def replace_dependencies_by_single_values_from_random_distribution(self,  node_ids: List[int],graph_id=0):
        """
        Replaces dependencies of specified nodes with single values drawn from a random distribution in a specific dependency controller.

        Args:
            node_ids (List[int]): List of node IDs to have their dependencies replaced.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.sampling_controller_sync = False
        self.dependency_controller[graph_id].replace_dependencies_by_single_values_from_random_distribution(node_ids=node_ids)

    def replace_dependencies_by_single_values_abstract_id(self, abstract_ids_and_values: List[Tuple[int, float]]):
        """
        Replaces dependencies of nodes identified by abstract IDs with single fixed values.

        Args:
            abstract_ids_and_values (List[Tuple[int, float]]): List of tuples containing abstract IDs and their corresponding values.
        """
        self.sampling_controller_sync = False
        for tuple in abstract_ids_and_values:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=tuple[0])
            graph_id, node_id = ids 
            node_id_and_value =   [[node_id,tuple[1]]]
            self.replace_dependencies_by_single_values(graph_id=graph_id, node_ids_and_values=node_id_and_value)

    def reset_dependencies_abstract_id(self, ids_abstract:List[int]):
        """
        Resets dependencies of nodes identified by abstract IDs, generating new dependencies.

        Args:
            ids_abstract (List[int]): List of abstract IDs corresponding to nodes.
        """
        self.sampling_controller_sync = False
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.dependency_controller[graph_id].reset_dependencies_specific(node_id=[node_id])
    
    

        
    def replace_dependencies_by_initial_distributions_abstract_id(self, ids_abstract:List[int]):
        """
        Replaces dependencies of nodes identified by abstract IDs with initial value distributions.

        Args:
            ids_abstract (List[int]): List of abstract IDs corresponding to nodes.
        """
        self.sampling_controller_sync = False
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.replace_dependencies_by_initial_distributions_abstract_id(node_id=[node_id],graph_id=graph_id)

    def replace_dependencies_by_single_random_values_abstract_id(self, ids_abstract:List[int]):
        """
        Replaces dependencies of nodes identified by abstract IDs with single random values drawn from a specified distribution.

        Args:
            ids_abstract (List[int]): List of abstract IDs corresponding to nodes.
        """
        self.sampling_controller_sync = False
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.replace_dependencies_by_single_values_from_random_distribution(node_id=[node_id],graph_id=graph_id)
            
            distribution = self.dependency_controller[graph_id].inputs_dependency.initial_value_distributions.get_distribution()
            self.dependency_controller[graph_id].se
        for source in self.list_of_sources:
            self.nodelist[source].dependency = distribution.get_value_from_distribution()

    def get_sourcelist_dependencymaker(self):
        """
        Retrieves a nested list of source nodes from all dependency controllers.

        Returns:
            List[List[int]]: A nested list containing lists of source nodes for each dependency controller.
        """
        nested_lists_of_sources = []
        for dependency_controller in self.dependency_controller:
            list_of_sources = dependency_controller.get_sources()
            nested_lists_of_sources.append(list_of_sources)
        return nested_lists_of_sources

    def set_sources_as_tsd_function(self):
        """
        Sets the source nodes in all dependency controllers to use time series data (TSD) functions as their dependency.
        """
        self.sampling_controller_sync = False
        for dependency_controller in self.dependency_controller:
            dependency_controller.set_sources_as_tsd_function()

    def set_sources_as_distributions(self):
        """
        Sets the source nodes in all dependency controllers to use specific distributions instead of a dependency.
        """
        self.sampling_controller_sync = False
        for dependency_controller in self.dependency_controller:
            dependency_controller.set_sources_as_distributions()

    def set_sources_to_fixed_values(self):
        """
        Sets the source nodes in all dependency controllers to fixed values instead of a dependency.
        """
        self.sampling_controller_sync = False
        for dependency_controller in self.dependency_controller:
            dependency_controller.set_sources_to_fixed_values()

    def get_nodelist_with_dependencies(self):
        """
        Retrieves the nodelists with dependencies from all dependency controllers.

        Returns:
            List[List[NodeObject]]: A nested list containing nodelists with dependencies for each dependency controller.
        """
        nested_nodelist = []
        for dependency_controller in self.dependency_controller:
           nodelist =  dependency_controller.get_nodelist_with_dependencies()
           nested_nodelist.append(nodelist)
        return nested_nodelist

    def show_dependencies(self,ids:List[int],resolution =100, graph_id=0 ): 
        """
        Displays the dependencies for specific nodes in a dependency controller.
        Visualizes the full spectrum: Function, Errorterm and Scatterplot .

        Args:
            ids (List[int]): List of node IDs to display dependencies for.
            resolution (int, optional): The resolution for plotting. Defaults to 100.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.dependency_controller[graph_id].show_dependencies(ids=ids,resolution=resolution)

    def show_dependencies_enforced_3d_visualisation(self,ids:List[int],resolution =100, graph_id=0, visualized_dimensions: Tuple[int, int] = (0, 1)): # only 2 dimensions shown with rest set to 0
        """
        same as show_dependencies but enforces 3d visualisation whenever possible

        Args:
            ids (List[int]): List of node IDs to display dependencies for.
            resolution (int, optional): The resolution for plotting. Defaults to 100.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
            visualized_dimensions (Tuple[int, int], optional): The dimensions to be visualized. Defaults to (0, 1).
        """
        self.dependency_controller[graph_id].show_dependencies_enforce_3d_plot(ids=ids,resolution=resolution, visualized_dimensions=visualized_dimensions)

    def show_dependency_functions_only(self,ids:List[int],resolution =100, graph_id=0 ):  
        """
        Displays only the deterministic dependency functions for specific nodes in a dependency controller.

        Args:
            ids (List[int]): List of node IDs to display dependency functions for.
            resolution (int, optional): The resolution for plotting. Defaults to 100.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.dependency_controller[graph_id].show_dependency_function(ids=ids,resolution=resolution)

    def show_dependency_errorterm_only(self,ids:List[int],resolution =100, graph_id=0 ):  
        """
        Displays only the error terms of the dependency for specific nodes in a dependency controller.

        Args:
            ids (List[int]): List of node IDs to display error terms for.
            resolution (int, optional): The resolution for plotting. Defaults to 100.
            graph_id (int, optional): The ID of the dependency controller. Defaults to 0.
        """
        self.dependency_controller[graph_id].show_dependency_errorterm(ids=ids,resolution=resolution)

    def _get_total_output_range(self):
        """
        Calculates the total output range across all dependency controllers.

        Returns:
            List[float]: Total output range as [minimum, maximum].
        """
        total_range = self.dependency_controller[0].inputs_dependency.range_of_output

        # Iterate through the rest of the controllers
        for controller in self.dependency_controller[1:]:
            current_range = controller.inputs_dependency.range_of_output
            total_range = [min(current_range[0], total_range[0]), 
                        max(current_range[1], total_range[1])]
        print(total_range)
        return total_range
            



# methods sampling controller
    
    @property
    def samples_raw(self):
        """
        Returns the raw sample data in the form of value-ID pairs.

        """

        samples = self.sampling_controller.return_samples_raw()
        return samples

    @property
    def samples_abstracted_id(self):
        """
        Returns the accumulated sample data sorted by node ID.
        """
        samples = self.sampling_controller.return_samples_abstracted_id()
        return samples

    @property
    def samples_accumulated_per_id(self):
        """
        Returns the sample data with abstracted IDs.

        """
        samples = self.sampling_controller.return_samples_accumulated_per_id()
        return samples

    def return_samples_abstracted_hidden_nodes(self,num_nodes_to_hide:int=5):
        """
        Returns the sample data with abstracted IDs.

        """
        samples = self.sampling_controller.return_samples_abstracted_hidden_nodes(num_nodes_to_hide=num_nodes_to_hide)
        return samples


    def load_full_nodelists_into_sampling_controller(self):
        """
        Loads all nodelists from the dependency controller into the sampling controller.
        first adjust the lenght of dependency_controlers, then load the nodelists"""
        full_nodelist = []
        for dependency_controller in self.dependency_controller:
            single_nodelist = dependency_controller.get_nodelist_with_dependencies()
            single_nodelist.sort()
            full_nodelist.append(single_nodelist)
        self.sampling_controller.load_nodelist_dependencies(nodelist_nested=full_nodelist)
        self.sampling_controller_sync = True

    def _check_and_update_sampling_controller_synchronization(self):
        """ checks if sampling and dependency controller are in sync if not 
        loads all nodelists again"""
        if self.sampling_controller_sync is False:
            self.load_full_nodelists_into_sampling_controller()

    def reset_config_sampler(self, config:Blueprint_sampling):
        """
        Resets the configuration of the sampling controller.

        Args:
            config (Blueprint_sampling): The new configuration settings to be applied.
        """
        self.sampling_controller.reset_config(config=config)

    def reset_samples(self):
        self.sampling_controller.reset_samples()

    def replace_id_shuffle_index(self):
        """
        Creates a new shuffle index for abstracting node IDs.
        """
        self._check_and_update_sampling_controller_synchronization()
        self.sampling_controller.replace_id_shuffle_index()
        
    def find_graph_id_and_node_id_from_shuffle_index(self,id_abstract:int):
        """
        finds real node id and graph id from abstracted node id
        """
        graph_id, node_id  =self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id_abstract)
        return graph_id, node_id 

    def sample_with_replaced_values(self, replaced_nodes:List[List[Tuple[int,float]]], number_of_samples:int=1):
        """
        samples while  replacing values for certain nodes provided by their real id.
        this method is usefull if you want to fix some nodes without actually changing the underlying depenencies.

        Args:
            replaced_nodes (List[Tuple[[int, float]]]): A list  of list of tuples (node_id, new_value) for nodes whose values are to be replaced, every upper list represents the graph component.
            number_of_samples(int): the amount of samples
        """
        self._check_and_update_sampling_controller_synchronization()
        self.sampling_controller.sample_with_replaced_values( replaced_nodes=replaced_nodes, number_of_samples=number_of_samples)
        self.sampling_controller.make_accumulated_samples()
        self.sampling_controller.make_abstracted_samples()
    
    def sample_with_replaced_values_abstract_ids(self, replaced_nodes_abstract_id:List[Tuple[int,float]], number_of_samples:int=1):
        """
        samples while  replacing values for certain nodes, provided by their abstracted id .
        this method is usefull if you want to fix some nodes without actually changing the underlying depenencies.

        Args:
            replaced_nodes (List[Tuple[int, float]]): A list of tuples (node_id, new_value) for nodes whose values are to be replaced
            number_of_samples(int): the amount of samples
        """
        self._check_and_update_sampling_controller_synchronization()
        self.sampling_controller.make_new_id_shuffle_index()
        nested_list_replaced_nodes= [[]]
        for replaced_node in replaced_nodes_abstract_id:
            graph_id, node_id = self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=replaced_node[0])
            id_value_pair = (node_id, replaced_node[1])

            # Ensure the nested_list_replaced_nodes is large enough
            while len(nested_list_replaced_nodes) <= graph_id:
                nested_list_replaced_nodes.append([])

            nested_list_replaced_nodes[graph_id].append(id_value_pair)

        self.sampling_controller.sample_with_replaced_values( replaced_nodes=nested_list_replaced_nodes, number_of_samples=number_of_samples)
        self.sampling_controller.make_accumulated_samples()
        self.sampling_controller.make_abstracted_samples()

    def sample_values_raw(self,number_of_samples=1):
        """
        Creates a list of value-ID pairs arrays for each sub-graph and saves it in the sampling controller, works with the real ids ."""

        self._check_and_update_sampling_controller_synchronization()
        self.sampling_controller.sample_value_id_pairs(number_of_samples=number_of_samples)
        self.sampling_controller.make_accumulated_samples()

    def sample_values_full(self,number_of_samples=1):
        """
        Creates a list of value-ID pairs arrays for each sub-graph and saves it in the sampling controller, adds a version with  abstracted-ids  ."""
        self._check_and_update_sampling_controller_synchronization()
        self.sample_values_raw(number_of_samples=number_of_samples)
        self.sampling_controller.make_abstracted_samples()

    def show_values_histogramm_abstracted(self, id_abstract:  Union[int, List[int]]):
        """
        Displays a histogram of sampled values for a specific node, or a list of nodes, provided by their abstract id.

        Args:
            id_abstract (Union[int, List[int]]): The ID of the graph component.

        """
        output_range = self._get_total_output_range()

        if isinstance( id_abstract, int):
            graph_id, node_id = self.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id_abstract)
            self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=node_id, output_range=output_range)

        elif isinstance(id_abstract, list):
            for id in id_abstract:
                graph_id, node_id = self.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
                self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=node_id, output_range=output_range)

    def print_samples_raw(self):
        """prints dataframe of samples"""
        self.sampling_controller.print_samples_raw()
    
    def show_values_histogramm_raw(self, node_id:  Union[int, List[int]], graph_id=0):
        """
        Displays a histogram of sampled values for a specific node.

        Args:
            graph_id (int): The ID of the graph component.
            node_id (Union[int, List[int]]): The ID of the node to visualize.
        """
        output_range = self._get_total_output_range()
        self.sampling_controller.show_values_histogramm(graph_id=graph_id,node_id=node_id, output_range=output_range)
        


        if isinstance( node_id, int):
            
            self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=node_id, output_range=output_range)

        elif isinstance(node_id, list):
            for id in node_id:
                self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=id, output_range=output_range)


    def export_value_id_samples_abstract(self, filename="last export"):
        """exports samples into a file, typically csv """
        self.sampling_controller.export_value_id_samples_abstract(filename=filename)

    def show_dependency_from_one_node(self, node_id_target:int, node_id_dependency:int, graph_id=0 , resolution = 100):
        """
        Visualizes the dependency of one node's value on another node's value.

        Args:
            node_id_target (int): The ID of the target node.
            node_id_dependency (int): The ID of the dependent node.
            range (Tuple[float, float]): The range of input values for the dependent node.
            graph_id (int, optional): The ID of the graph component. Defaults to 0.
            resolution (int, optional): The resolution for the plot. Defaults to 100.
        """
        self._check_and_update_sampling_controller_synchronization()
        self.load_full_nodelists_into_sampling_controller()
        range = self.dependency_controller[graph_id].inputs_dependency.range_of_output
        self.sampling_controller.show_dependency_from_one_node(node_id_target=node_id_target, node_id_dependency=node_id_dependency, graph_id=graph_id,range=range , resolution=resolution)
    
    def show_dependency_from_2_nodes(self, node_id_target:int, node_id_dependency_x:int,node_id_dependency_y:int,  graph_id=0 , resolution = 10):
        """
        Visualizes the dependency of one node's value on the values of two other nodes.

        Args:
            node_id_target (int): The ID of the target node.
            node_id_dependency_x (int): The ID of the first dependent node.
            node_id_dependency_y (int): The ID of the second dependent node.
            graph_id (int, optional): The ID of the graph component. Defaults to 0.
            resolution (int, optional): The resolution for the 3D plot. Defaults to 10.
        """
        self._check_and_update_sampling_controller_synchronization()
        range_f = self.dependency_controller[graph_id].inputs_dependency.range_of_output
        self.sampling_controller.show_dependency_from_2_nodes(node_id_target=node_id_target, node_id_dependency_x= node_id_dependency_x,node_id_dependency_y= node_id_dependency_y,  graph_id= graph_id , resolution = resolution, range_f=range_f)       
   
    def show_dependency_from_parents_scatterplot(self, node_id_target:int, graph_id:int=0, resolution:int=100, visualized_dimensions:Tuple[int,int] = (0,1)):
        """
        Visualizes the dependency of a node's value on its parent nodes' values using a scatter plot.

        Args:
            node_id_target (int): The ID of the target node.
            graph_id (int, optional): The ID of the graph component. Defaults to 0.
            resolution (int, optional): The resolution for the plot. Defaults to 100.
            visualized_dimensions (Tuple[int, int], optional): The dimensions to visualize in the scatter plot. Defaults to (0, 1).
        """
        self._check_and_update_sampling_controller_synchronization()
        range_f = self.dependency_controller[graph_id].inputs_dependency.range_of_output
        self.sampling_controller.show_dependency_from_parents_scatterplot(node_id_target= node_id_target, graph_id= graph_id, resolution= resolution, visualized_dimensions= visualized_dimensions, range_f=range_f )

#serialization methods

    def reset_config_main(self, config:Blueprint_main_controller):
        """
        Resets the main configuration of the controller.

        Args:
            config (Blueprint_main_controller): The new configuration to be set for the controller.
        """
        self.config = config

    def safe_dataset(self, file_path=None, file_name="last_dataset"):
        """
        Saves the current dataset/samples_raw to a file typically pickle.

        Args:
            file_path (str, optional): The file path where the dataset will be saved. If None, a default path is used.
            file_name (str, optional): The file name for the saved dataset. Defaults to "last_dataset".
        """
        dataset = self.sampling_controller.return_samples_raw()
        self.config.object_serializer.set_object(object=dataset)
        self.config.object_serializer.safe_object(file_path=file_path,file_name=file_name)

    def load_dataset(self, file_path=None, file_name="last_dataset") :
        """
        Loads a dataset from a file.

        Args:
            file_path (str, optional): The file path from where the dataset will be loaded. If None, a default path is used.
            file_name (str, optional): The file name of the dataset to be loaded. Defaults to "last_dataset".
        """
        self.config.object_serializer.load_object(file_path=file_path,file_name=file_name)
        dataset = self.config.object_serializer.return_object()
        self.sampling_controller.id_value_list_of_arrays = dataset

    def safe_controller_state(self, file_path=None, file_name="test"):
        """
        Saves the current state of the controller to a file.

        Args:
            file_path (str, optional): The file path where the controller's state will be saved. If None, a default path is used.
            file_name (str, optional): The file name for the saved state. Defaults to "test".
        """
        self.config.object_serializer.set_object(object=self)
        self.config.object_serializer.safe_object(file_path=file_path,file_name=file_name)
    
    @classmethod
    def load_controller_state(cls,config:Blueprint_main_controller, file_path=None, file_name="test") -> "Cg_flex_controller":
        """
        Class method to load the state of a controller from a file.

        Args:
            config (Blueprint_main_controller): The configuration for the controller.
            file_path (str, optional): The file path from where the controller's state will be loaded. If None, a default path is used.
            file_name (str, optional): The file name of the state to be loaded. Defaults to "test".

        Returns:
            Cg_flex_controller: The loaded controller.
        """
        serializer = config.object_serializer
        serializer.load_object(file_path=file_path,file_name=file_name)
        object = serializer.return_object()
        return object

    def safe_nodelists_dependencies_to_dict(self, file_path=None): 
        """Saves the nodelist within the dependency controller including its nested dependency object to a file. """
        counter = 1
        for dependency_controller in self.dependency_controller:
            dependency_controller.safe_nodelist_with_dependencies_pikle_gpy_models_to_dict(file_name=f"default_dependencysafe_graphcomponent_nr{counter}", file_path=file_path)
            counter += 1

    def load_nodelists_dependencies_from_dict(self, file_path=None):
        """ Loads the nodelist within the dependency controller including its nested dependency object from a file. loading the nodelist within the dependency_controller including its nested dependencyobject"""
        counter = 1
        for dependency_controller in self.dependency_controller:
            dependency_controller.load_nodelist_with_dependencies_pikle_gpy_models_from_dict(file_name=f"default_dependencysafe_graphcomponent_nr{counter}", file_path=file_path)
            counter += 1

    def safe_nodelists_graph(self, file_path=None):  
        """Saves the nodelists for graphs individually to a file.lets user to safe nodelists for graphs individually """
        counter = 0
        for graph in self.graph_controller:
            graph.safe_nodelist_pikle(file_name=f"default_grahphsafe_graphcomponent_nr{counter}", file_path=file_path)
            counter += 1
    
    def load_nodelists_graph(self, file_path=None): 
        """Loads the nodelists for graphs individually from a file. Lets user load nodelists for graphs individually"""
        counter = 0
        for graph in self.graph_controller:
            graph.load_nodelist_pikle(file_name=f"default_grahphsafe_graphcomponent_nr{counter}",file_path=file_path)
            counter += 1

  