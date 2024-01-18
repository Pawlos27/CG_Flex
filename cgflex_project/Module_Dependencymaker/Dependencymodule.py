from cgflex_project.Shared_Classes.blueprints import Blueprint_dependency
from  cgflex_project.Module_Dependencymaker._dependencymaker import Dependencies
from typing import Any, List, Type, Tuple, Optional
from cgflex_project.Shared_Classes.NodeObject import Nodeobject
from cgflex_project.Shared_Classes.distributions import IDistributions
import os
import pickle
import GPy
from  cgflex_project.Module_Dependencymaker._errortermmaker_collection import Error_distribution_normal_variable_variance
from  cgflex_project.Module_Dependencymaker._dependencymaker_tsd_functions import ITsd_functions



class Dependency_controller:
    """
    Manages and controls the dependencies of nodes in a NodeList.

    The Dependency_controller is one od the 3 Sub controllers of the Framework. It is responsible for creating, resetting, and manipulating the dependencies of nodes based on a given configuration. 
    It offers various methods to manipulate dependencies, such as setting them to fixed values, replacing them with distributions, or using TSD functions.
    It can be used in tandem with the sampling controller, manipulating dependencies between sampling rounds.

    Args:
        config (Blueprint_dependency): The configuration object used to initialize dependency settings.

    Attributes:
        nodelist (List[Type[Nodeobject]]): The list of nodes whose dependencies are managed by the controller, representing a graph structure.
        list_of_sources (List[int]): The list of the node_ids of source nodes in the nodelist.
    """
    def __init__(self, config: Blueprint_dependency):
        """
        Initializes the Dependency_controller instance with the given configuration.
        """        
        self.inputs_dependency = config
        self.nodelist = None
        self.list_of_sources = None

    def load_nodelist_graph(self, nodelist: List[Type[Nodeobject]]): 
        """
        Loads a NodeList/Graph into the controller.

        Args:
            nodelist (List[Type[Nodeobject]]): The NodeList to be managed by the controller.
        """
        self.nodelist = nodelist
        
    def make_dependencies(self):    # sources are set as fixed values by default
        """
        Creates dependencies for all nodes in the NodeList.

        This method sets dependencies for each node in the nodelist, considering whether nodes are sources or have parents it sets
        the dependency of sources to be fixed values by default, can be also exchanged to a distribution or a tsd function later on.
        """
        self.nodelist.sort()    
        self._set_list_of_sources() 
        self.set_sources_to_fixed_values()
        for node in self.nodelist:
            if len(node.parents) >0 :
                node.dependency = self.inputs_dependency.dependency_setter.set_dependencies(node=node, range_of_output= self.inputs_dependency.range_of_output)
            else:
                pass

    def reset_config(self,inputs_dependency: Blueprint_dependency):
        """
        Resets the configuration used for setting dependencies.

        Args:
            inputs_dependency (Blueprint_dependency): The new configuration to be used.
        """
        self.inputs_dependency= inputs_dependency

    def _make_list_of_sources(self)-> List[int]:
        list_of_sources = []
        for node in self.nodelist:
            if node.source == True:
                list_of_sources.append(node.id)
        return list_of_sources

    def _set_list_of_sources(self):
        self.list_of_sources = self._make_list_of_sources()

    def get_sources(self)->List[int]:
        """
        Retrieves the list of source nodes.

        Raises:
            ValueError: If no NodeList is loaded into the controller.
        """
        if self.list_of_sources == None:
            raise ValueError("no nodelist! load nodelist into dependencycontroller before trying to access a list of sources")
        else: 
            return self.list_of_sources

    def reset_dependencies_specific(self,node_ids: List[int]):
        """
        Resets dependencies for specific nodes based on their IDs.
        
        Args:
            node_ids (List[int]): List of node IDs to reset dependencies for.
        """
        for node_id in node_ids:
            self.nodelist[node_id].dependency = self.inputs_dependency.dependency_setter.set_dependencies(node=self.nodelist[node_id])

    def replace_dependencies_by_initial_distributions(self,node_ids: List[int]):
        """
        Replaces dependencies for specific nodes with initial probability distributions.

        Args:
            node_ids (List[int]): List of node IDs to replace dependencies for.
        """
        for node_id in node_ids:
            self.nodelist[node_id].dependency = self.inputs_dependency.initial_value_distributions.get_distribution()

    def replace_all_dependencies_by_initial_distributions(self):
        """Replaces all dependencies in the NodeList with initial probability distributions."""
       
        for node in self.nodelist:
            node.dependency = self.inputs_dependency.initial_value_distributions.get_distribution()

    def replace_dependencies_by_single_values(self,node_ids_and_values: List[Tuple[int, float]]):
        """
        Replaces dependencies for specific nodes with single values.

        Args:
            node_ids_and_values (List[Tuple[int, float]]): List of tuples containing node IDs and values to set.
  
        """
        for tuple in node_ids_and_values:
      
            self.nodelist[tuple[0]].dependency = tuple[1]
    
    def replace_dependencies_by_single_values_from_random_distribution(self,node_ids: List[int]):
        """
        Replaces dependencies for specific nodes with values from a random distribution.

        Args:
            node_ids (List[int]): List of node IDs to replace dependencies for.
        """
        distribution = self.inputs_dependency.initial_value_distributions.get_distribution()
        for node_id in node_ids:
            self.nodelist[node_id].dependency = distribution.get_value_from_distribution()
    
    def replace_dependencies_by_tsd_function(self,node_ids: List[int]):
        """
        Replaces dependencies for specific nodes with a TSD function.

        Args:
            node_ids (List[int]): List of node IDs to replace dependencies for.
        """
        for node_id in node_ids:
            tsd_function = self.inputs_dependency.tsd_collection.get_tsd_function()
            self.nodelist[node_id].dependency = tsd_function

    def set_sources_as_tsd_function(self):
        """Sets all source nodes' dependencies as TSD functions."""
       
        for source in self.list_of_sources:
            tsd_function = self.inputs_dependency.tsd_collection.get_tsd_function()
            self.nodelist[source].dependency = tsd_function

    def set_sources_as_distributions(self):
        """Sets all source nodes' dependencies as probability distributions."""
       
        for source in self.list_of_sources:
            self.nodelist[source].dependency = self.inputs_dependency.initial_value_distributions.get_distribution()

    def set_sources_to_fixed_values(self):
        """Sets all source nodes' dependencies to fixed values."""
       
        distribution = self.inputs_dependency.initial_value_distributions.get_distribution()
        for source in self.list_of_sources:
            self.nodelist[source].dependency = distribution.get_value_from_distribution()

    def get_nodelist_with_dependencies(self):
        """
        Retrieves the NodeList with dependencies set.

        Raises:
            ValueError: If there is no graph or no dependencies set in the NodeList.
        """
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph and add dependencies")
        elif self.nodelist[0].dependency == None:
            raise ValueError("there are no dependencies in the nodelist yet, please add dependencies")
        else:
            return self.nodelist

    def show_dependencies(self,ids:List[int],resolution:int):
        """
        Displays the dependencies for specific nodes.

        Args:
            ids (List[int]): List of node IDs to show dependencies for.
            resolution (int): The resolution for plotting dependencies.
        """ 
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, (float, int)):
                        print(f" Node_ID: {id}   Dependency is a fixed Value :  {node.dependency} ")
                    elif isinstance(node.dependency, IDistributions):
                        print(f" Node_ID: {id}    Dependency is a Distribution")
                        node.dependency.plot_distribution(label=f"distribution for node-id {id}")
                    elif isinstance(node.dependency, Dependencies):
                        print(f" Node_ID: {id}   Dependency is a Dependency_Object")
                        node.dependency.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=resolution, label=f" node_id ={id}   Function")
                        node.dependency.function_model.show_function_borders(node_id=node.id)
                        node.dependency.errorterm_model.show_error_distribution(label =f" node_id ={id}   ")
                    elif isinstance(node.dependency, ITsd_functions):
                        node.dependency.plot_function(label=f"tsd function for node-id ={id}")

    def show_dependencies_enforce_3d_plot(self,ids:List[int],resolution:int, visualized_dimensions: Tuple[int, int] = (0, 1)): 
        """
        Displays dependencies for specific nodes with an enforced 3D plot.

        Args:
            ids (List[int]): List of node IDs to show dependencies for.
            resolution (int): The resolution for plotting dependencies.
            visualized_dimensions (Tuple[int, int]): Dimensions to be visualized in the plot.
        """
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, (float, int)):
                        print(f" Node_ID: {id}   Dependency  is a fixed Value :  {node.dependency} ")
                    elif isinstance(node.dependency, IDistributions):
                        print(f" Node_ID: {id}   Dependency  is a Distribution")
                        node.dependency.plot_distribution()
                    elif isinstance(node.dependency, Dependencies):
                        print(f" Node_ID: {id}   Dependency is a Dependency_Object")
                        node.dependency.function_model.show_functions_3d_plot_when_possible(resolution=resolution, label=f" node_id ={id}   Function", visualized_dimensions=visualized_dimensions)
                        node.dependency.function_model.show_function_borders(node_id=node.id)
                        node.dependency.errorterm_model.show_error_distribution(label =f" node_id ={id}   ")

    def show_dependency_function(self,ids:List[int],resolution:int): 
        """
        Displays only the dependency functions for specific nodes.

        Args:
            ids (List[int]): List of node IDs to display dependency functions for.
            resolution (int): The resolution for plotting functions.
        """
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=resolution)
                    else:
                        print("no functions in dependency for {id}")

    def show_dependency_function_enforce_3d_plot(self,ids:List[int],resolution:int,visualized_dimensions: Tuple[int, int] = (0, 1)): 
        """
        Displays only the dependency functions for specific nodes (enforces 3D plot).

        Args:
            ids (List[int]): List of node IDs to display dependency functions for.
            resolution (int): The resolution for plotting functions.
            visualized_dimensions (Tuple[int, int]): Dimensions to be visualized in the plot.
        """
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_when_possible(resolution=resolution, visualized_dimensions= visualized_dimensions)
                    else:
                        print("no functions in dependency for {id}")
    def show_dependency_errorterm(self,ids:List[int],resolution:int): 
        """
        Displays only the error terms in the dependencies for specific nodes.

        Args:
            ids (List[int]): List of node IDs to display error terms for.
            resolution (int): The resolution for plotting error terms.
        """
        for id in ids:    
            for node in self.nodelist:
                if node.id == id:                  
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=resolution)
                    else:
                        print("no errorterm in dependency for {id}")

    def show_dependency_errorterm_enforce_3d_plot(self,ids:List[int],resolution =30): 
        """
        Displays the error terms in the dependencies for specific nodes( with 3d plot for the sigma function in conditional dependencies).

        Args:
            ids (List[int]): List of node IDs to display error terms for.
            resolution (int): The resolution for plotting error terms.
        """
        for id in ids:    
            for node in self.nodelist:
                if node.id == id:                  
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_when_possible(resolution=resolution)
                    else:
                        print("no errorterm in dependency for {id}")

    def print_kernels_used(self,ids:List[int]): 
        """
        Prints the kernels used in the dependencies for specific nodes.

        Args:
            ids (List[int]): List of node IDs to print kernels for.
        """
        pass

    def safe_nodelist_with_dependencies_pikle_gpy_models_to_dict(self,file_name:str, file_path:Optional[str]):
        """ NOT USED IN MAIN CONTROLLER 
        Saves the NodeList with dependencies as a pickle file, converting GPy models to dictionaries.

        Args:
            file_name (str): The name of the file to save the NodeList.
            file_path (Optional[str]): The path where the file will be saved. If None, defaults to a predefined data folder.
        
        Raises:
            ValueError: If there is no NodeList to save.
        """
        if self.nodelist is not None:
            dicted_nodelist= self._make_nodelist_with_Gpy_to_dict(nodelist=self.nodelist)


            if file_path == None:
                data_folder = os.path.join(os.path.dirname(__file__), 'data')
                file_path = os.path.join(data_folder, file_name)

            with open(file_path, 'wb') as file:
                pickle.dump(dicted_nodelist, file)
        else:
            raise ValueError("No data to save. Please set data using set_data method first.")
        
    def load_nodelist_with_dependencies_pikle_gpy_models_from_dict(self,file_name:str,file_path:Optional[str]):
        """ NOT USED IN MAIN CONTROLLER 
        Loads the NodeList with dependencies from a pickle file, converting dictionaries back to GPy models.

        Args:
            file_name (str): The name of the file to load the NodeList from.
            file_path (Optional[str]): The path where the file is located. If None, defaults to a predefined data folder.
        """
        if file_path == None:
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'rb') as file:
            dicted_nodelist = pickle.load(file)
        
        self.nodelist = self._make_nodelist_with_Gpy_from_dict(nodelist=dicted_nodelist)

    def safe_nodelist_pikle(self,file_name:str, file_path:Optional[str]):
        """ NOT USED IN MAIN CONTROLLER 
        Saves the NodeList to a pickle file.

        Args:
            file_name (str): The name of the file to save the NodeList.
            file_path (Optional[str]): The path where the file will be saved. If None, defaults to a predefined data folder.
        
        Raises:
            ValueError: If there is no NodeList to save.
        """
        if self.nodelist is not None:
            if file_path == None:
                data_folder = os.path.join(os.path.dirname(__file__), 'data')
                file_path = os.path.join(data_folder, file_name)

            with open(file_path, 'wb') as file:
                pickle.dump(self.nodelist, file)
        else:
            raise ValueError("No data to save. Please set data using set_data method first.")
        
    def load_nodelist_pikle(self,file_name:str,file_path:Optional[str]):
        """ NOT USED IN MAIN CONTROLLER 
        Loads the NodeList from a pickle file.

        Args:
            file_name (str): The name of the file to load the NodeList from.
            file_path (Optional[str]): The path where the file is located. If None, defaults to a predefined data folder.
        """
        if file_path == None:
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'rb') as file:
            self.nodelist = pickle.load(file)

    def _make_nodelist_with_Gpy_to_dict(nodelist: List[Type[Nodeobject]])->List[Type[Nodeobject]]:
        dicted_nodelist=nodelist
        counter=0
        for node in dicted_nodelist:
        
            if isinstance(node.dependency, Dependencies):
                counter +=1
                print(f"durchlauf {counter}  davor")
                print(f"id of node is {node.id}")
                functionlist = node.dependency.function_model.functions
                for function in functionlist:
                    print(function.function_model.kern)
                    dicted_model = function.function_model.to_dict()
                    function.function_model = dicted_model
                print(f"durchlauf {counter}  dannach")
                if isinstance(node.dependency.errorterm_model, Error_distribution_normal_variable_variance):
                    print(f"durchlauf {counter}  errorterm davor" )
                    errorterm_functionlist = node.dependency.errorterm_model.function_model.functions
                    for function in errorterm_functionlist:
                        dicted_model = function.function_model.to_dict()
                        function.function_model = dicted_model
                    print(f"durchlauf {counter}  errorterm dannach" )
        return dicted_nodelist
    
    def _make_nodelist_with_Gpy_from_dict(nodelist: List[Type[Nodeobject]])->List[Type[Nodeobject]]:
        undicted_nodelist=nodelist
        for node in undicted_nodelist:
            if isinstance(node.dependency, Dependencies):
                functionlist = node.dependency.function_model.functions
                for function in functionlist:
                    undicted_model = GPy.models.GPRegression.from_dict(function.function_model)
                    function.function_model = undicted_model
            
                if isinstance(node.dependency.errorterm_model, Error_distribution_normal_variable_variance):
                    errorterm_functionlist = node.dependency.errorterm_model.function_model.functions
                    for function in errorterm_functionlist:
                        undicted_model = GPy.models.GPRegression.from_dict(function.function_model)
                        function.function_model = undicted_model
        return undicted_nodelist
        



if __name__ == "__main__":
    pass
 