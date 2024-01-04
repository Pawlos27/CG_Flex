
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
    def __init__(self,config= Blueprint_main_controller):
        """_summary_

        Args:
            config (_type_, optional): _description_. Defaults to Blueprint_main_controller.
        """
        
        
        self.graph_controller = None
        self.dependency_controller = None
        self.sampling_controller = None
        self.nodelist = None
        self.id_value_arrays = None
        self.samples_raw = None
        self.samples_abstracted_id = None
        self.samples_accumulated_per_id_raw = None
        self.id_shuffle_index = None
        self.config = config




# methods graph controller 
    def reset_controller(self, size):
        list_of_graph_controller = self.config.controller_coordinator.make_list_graph_controller(size=size)
        self.graph_controller = list_of_graph_controller
        list_of_dependency_controller = self.config.controller_coordinator.make_list_dependency_controller(size=size)
        self.dependency_controller = list_of_dependency_controller
        self.sampling_controller = self.config.sampling_controller

    def get_number_of_graph_components(self):
        components = len(self.graph_controller)
        return components
    
    def make_graphs(self, number_of_decoupled_elements:int):
        self.reset_controller(size=number_of_decoupled_elements) 
        for graph_controller in self.graph_controller:
            graph_controller.make_graph()

    def reset_configs_graph(self, config: Blueprint_graph, graph_id:int,):
        self.graph_controller[graph_id].reset_config(config=config)

    def reset_layers_graph(self, graph_id=0):
        self.graph_controller[graph_id].reset_layers()

    def set_new_sources_graph(self, graph_id=0):
        self.graph_controller[graph_id].new_sources()

    def set_new_sinks_graph(self, graph_id=0):
        self.graph_controller[graph_id].new_sinks()

    def set_new_sinks_and_sources_graph(self, graph_id=0):
        self.graph_controller[graph_id].new_sinks_and_sources()

    def set_new_edges_graph(self, graph_id=0):
        self.graph_controller[graph_id].new__edges()

    def print_nodelists_graph(self,  graph_ids: Optional[List[int]] = None): # can print nodelists of selected graphs
        if graph_ids == None:
            for graph_controller in self.graph_controller:
                graph_controller.print_nodelist()
        else:
            for graph_id in graph_ids:
                self.graph_controller[graph_id].print_nodelist()

    def get_verticelist_graph(self): # returns nested verticelist for all graphs
        nested_verticelist = []
        for graph_controller in self.graph_controller:
            verticelist = graph_controller.get_verticelist()
            nested_verticelist.append(verticelist)
        return nested_verticelist

    def get_sourcelist_graph(self): # returns nested sourcelist for all graphs
        nested_sourcelist = []
        for graph_controller in self.graph_controller:
            sourcelist = graph_controller.get_list_of_sources()
            nested_sourcelist.append(sourcelist)
        return nested_sourcelist
    
    def get_sinklist_graph(self, node_id: int, graph_id:int): #returns nested sinklist for all graphs
        nested_sinklist = []
        for graph_controller in self.graph_controller:
            sinklist = graph_controller.get_list_of_sinks()
            nested_sinklist.append(sinklist)
        return nested_sinklist
    
    def get_nodelists_graph(self, node_id: int, graph_id:int): # returns nested nodelists of al lgraphs
        nested_nodelists = []
        for graph_controller in self.graph_controller:
            nodelist = graph_controller.get_nodelist_graph()
            nested_nodelists.append(nodelist)
        return nested_nodelists

    def plot_graph(self,graph_ids: Optional[List[int]] = None): # plots all graphs or graphs with listed id´s
        if graph_ids == None:
            counter = 0
            for graph in self.graph_controller:
                graph.showgraph(plot_title= f"DAG_Graph Nr:{counter}")
                counter += 1
        else:
            for graph_id in graph_ids:
                    graph_title = f"DAG_Graph Nr:{graph_id}"
                    self.graph_controller[graph_id].showgraph(plot_title=graph_title)

    def plot_graph_by_layer(self,graph_ids: Optional[List[int]] = None): # plots all graphs or graphs with listed id´s
        if graph_ids == None:
            counter = 0
            for graph in self.graph_controller:
                graph.showgraph_layer_perspective(plot_title= f"DAG_Graph by Layer Nr:{counter}")
                counter += 1
        else:
            for graph_id in graph_ids:
                    graph_title = f"DAG_Graph by Layer Nr:{graph_id}"
                    self.graph_controller[graph_id].showgraph_layer_perspective(plot_title=graph_title)
        

    def get_nodelists_graph(self):
        nested_nodelist = []
        for controller in self.graph_controller:
           nodelist =  controller.get_nodelist_graph()
           nested_nodelist.append(nodelist)
        return nested_nodelist



# Methods dependencycontroller 
    def load_graph_nodelists_into_dependency_controller(self): # first adjust the lenght of dependency_controlers, then load the nodelists from the graph_controller
        for i in range (len(self.graph_controller)):
            self.dependency_controller[i].load_nodelist_graph(nodelist=self.graph_controller[i].get_nodelist_graph())

    def make_dependencies(self):
        self.load_graph_nodelists_into_dependency_controller()
        for dependency_controller in self.dependency_controller:
            dependency_controller.make_dependencies()
    
    def reset_dependencies(self):
        for dependency_controller in self.dependency_controller:
            dependency_controller.make_dependencies()

    def reset_configs_dependency(self, config: Blueprint_dependency, graph_id=0):
        self.dependency_controller[graph_id].reset_config(config=config)

    def reset_dependencies_specific(self, node_ids: List[int], graph_id=0): 
        self.dependency_controller[graph_id].reset_dependencies_specific(node_id=node_ids)
    
    def reset_tsd_counter(self):
        self.sampling_controller.reset_tsd_counter()
    
    def replace_dependencies_by_tsd_function(self, node_ids: List[int], graph_id=0): 
        self.dependency_controller[graph_id].replace_dependencies_by_tsd_function(node_id=node_ids)

    def replace_dependencies_by_tsd_function_abstract_id(self, ids_abstract:List[int]):
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.dependency_controller[graph_id].replace_dependencies_by_tsd_function(node_id=[node_id])

    def replace_dependencies_by_initial_distributions(self, node_ids: List[int], graph_id=0):
        self.dependency_controller[graph_id].replace_dependencies_by_initial_distributions(node_ids=node_ids)

    def replace_all_dependencies_by_initial_distributions(self):
        for dependency_controller in self.dependency_controller:
            dependency_controller.replace_all_dependencies_by_initial_distributions()

    def replace_dependencies_by_single_values(self, node_ids_and_values: List[Tuple[int, float]] ,graph_id=0):
        self.dependency_controller[graph_id].replace_dependencies_by_single_values(node_ids_and_values=node_ids_and_values)

    def replace_dependencies_by_single_values_from_random_distribution(self,  node_ids: List[int],graph_id=0):
        self.dependency_controller[graph_id].replace_dependencies_by_single_values_from_random_distribution(node_ids=node_ids)

    def replace_dependencies_by_single_values_abstract_id(self, abstract_ids_and_values: List[Tuple[int, float]]):
        for tuple in abstract_ids_and_values:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=tuple[0])
            graph_id, node_id = ids 
            node_id_and_value =   [[node_id,tuple[1]]]
            self.replace_dependencies_by_single_values(graph_id=graph_id, node_ids_and_values=node_id_and_value)

    def reset_dependencies_abstract_id(self, ids_abstract:List[int]):
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.dependency_controller[graph_id].reset_dependencies_specific(node_id=[node_id])
    
    

        
    def replace_dependencies_by_initial_distributions_abstract_id(self, ids_abstract:List[int]):
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.replace_dependencies_by_initial_distributions_abstract_id(node_id=[node_id],graph_id=graph_id)

    def replace_dependencies_by_single_random_values_abstract_id(self, ids_abstract:List[int]): # hier prüfen!!!!!
        for id in ids_abstract:
            ids= self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
            graph_id, node_id = ids 
            self.replace_dependencies_by_single_values_from_random_distribution(node_id=[node_id],graph_id=graph_id)
            
            distribution = self.dependency_controller[graph_id].inputs_dependency.initial_value_distributions.get_distribution()
            self.dependency_controller[graph_id].se
        for source in self.list_of_sources:
            self.nodelist[source].dependency = distribution.get_value_from_distribution()

    def get_sourcelist_dependencymaker(self):
        nested_lists_of_sources = []
        for dependency_controller in self.dependency_controller:
            list_of_sources = dependency_controller.get_sources()
            nested_lists_of_sources.append(list_of_sources)
        return nested_lists_of_sources

    def set_sources_as_tsd_function(self):
        for dependency_controller in self.dependency_controller:
            dependency_controller.set_sources_as_tsd_function()

    def set_sources_as_distributions(self):
        for dependency_controller in self.dependency_controller:
            dependency_controller.set_sources_as_distributions()

    def set_sources_to_fixed_values(self):
        for dependency_controller in self.dependency_controller:
            dependency_controller.set_sources_to_fixed_values()

    def get_nodelist_with_dependencies(self):
        nested_nodelist = []
        for dependency_controller in self.dependency_controller:
           nodelist =  dependency_controller.get_nodelist_with_dependencies()
           nested_nodelist.append(nodelist)
        return nested_nodelist

    def show_dependencies(self,ids:List[int],resolution =100, graph_id=0 ): 
        self.dependency_controller[graph_id].show_dependencies(ids=ids,resolution=resolution)

    def show_dependencies_enforced_3d_visualisation(self,ids:List[int],resolution =100, graph_id=0): # only 2 dimensions shown with rest set to 0
        self.dependency_controller[graph_id].show_dependencies_enforce_3d_plot(ids=ids,resolution=resolution)

    def show_dependency_functions_only(self,ids:List[int],resolution =100, graph_id=0 ):  
        self.dependency_controller[graph_id].show_dependency_function(ids=ids,resolution=resolution)

    def show_dependency_errorterm_only(self,ids:List[int],resolution =100, graph_id=0 ):  
        self.dependency_controller[graph_id].show_dependency_errorterm(ids=ids,resolution=resolution)

    def _get_total_output_range(self):
        # Initialize total_range with the first controller's range
        total_range = self.dependency_controller[0].inputs_dependency.range_of_output

        # Iterate through the rest of the controllers
        for controller in self.dependency_controller[1:]:
            current_range = controller.inputs_dependency.range_of_output
            total_range = [min(current_range[0], total_range[0]), 
                        max(current_range[1], total_range[1])]
        print(total_range)
        return total_range
            



# methods samling controller
    def load_full_nodelists_into_sampling_controller(self): # first adjust the lenght of dependency_controlers, then load the nodelists
        full_nodelist = []
        for dependency_controller in self.dependency_controller:
            single_nodelist = dependency_controller.get_nodelist_with_dependencies()
            single_nodelist.sort()
            full_nodelist.append(single_nodelist)
        self.sampling_controller.load_nodelist_dependencies(nodelist_nested=full_nodelist)

    def reset_config_sampler(self, config:Blueprint_sampling):  
        self.sampling_controller.reset_config(config=config)

    def replace_id_shuffle_index(self):
        self.sampling_controller.replace_id_shuffle_index()
        
    def find_graph_id_and_node_id_from_shuffle_index(self,id_abstract:int):
        graph_id, node_id  =self.sampling_controller.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id_abstract)
        return graph_id, node_id 

    def sample_values_raw(self,number_of_samples=1):
        self.load_full_nodelists_into_sampling_controller()
        self.sampling_controller.sample_value_id_pairs(number_of_samples=number_of_samples)
        self.sampling_controller.make_accumulated_samples()
        self.samples_raw = self.sampling_controller.return_id_value_list_of_arrays()
        self.samples_accumulated_per_id_raw = self.sampling_controller.return_samples_accumulated_per_id()

    def sample_values_full(self,number_of_samples=1):
        self.sample_values_raw(number_of_samples=number_of_samples)
        self.sampling_controller.make_abstracted_samples()
        self.samples_abstracted_id = self.sampling_controller.return_samples_abstracted_id()

    def show_values_histogramm_abstracted(self, id_abstract:  Union[int, List[int]]):
        output_range = self._get_total_output_range()

        if isinstance( id_abstract, int):
            graph_id, node_id = self.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id_abstract)
            self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=node_id, output_range=output_range)

        elif isinstance(id_abstract, list):
            for id in id_abstract:
                graph_id, node_id = self.find_graph_id_and_node_id_from_shuffle_index(id_abstract=id)
                self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=node_id, output_range=output_range)


    
    def show_values_histogramm_raw(self, node_id:  Union[int, List[int]], graph_id=0):
        output_range = self._get_total_output_range()
        self.sampling_controller.show_values_histogramm(graph_id=graph_id,node_id=node_id, output_range=output_range)
        


        if isinstance( node_id, int):
            
            self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=node_id, output_range=output_range)

        elif isinstance(node_id, list):
            for id in node_id:
                self.sampling_controller.show_values_histogramm(graph_id= graph_id, node_id=id, output_range=output_range)


    def export_value_id_samples_abstract(self, filename="last export"):
        self.sampling_controller.export_value_id_samples_abstract(filename=filename)

    
    def show_dependency_from_one_node(self, node_id_target:int, node_id_dependency:int, graph_id=0 , resolution = 100):

        self.load_full_nodelists_into_sampling_controller()
        range = self.dependency_controller[graph_id].inputs_dependency.range_of_output
        self.sampling_controller.show_dependency_from_one_node(node_id_target=node_id_target, node_id_dependency=node_id_dependency, graph_id=graph_id,range=range , resolution=resolution)

    
    def show_dependency_from_2_nodes(self, node_id_target:int, node_id_dependency_x:int,node_id_dependency_y:int,  graph_id=0 , resolution = 10):

        self.load_full_nodelists_into_sampling_controller()
        range_f = self.dependency_controller[graph_id].inputs_dependency.range_of_output
        self.sampling_controller.show_dependency_from_2_nodes(node_id_target=node_id_target, node_id_dependency_x= node_id_dependency_x,node_id_dependency_y= node_id_dependency_y,  graph_id= graph_id , resolution = resolution, range_f=range_f)       



#serialization methods

    def reset_config_main(self, config:Blueprint_main_controller):
        self.config = config

    def safe_dataset(self, file_path=None, file_name="last_dataset"):
        self.config.object_serializer.set_object(object=self.id_value_arrays)
        self.config.object_serializer.safe_object(file_path=file_path,file_name=file_name)

    def load_dataset(self, file_path=None, file_name="last_dataset") :
        self.config.object_serializer.load_object(file_path=file_path,file_name=file_name)
        self.id_value_arrays

    def safe_controller_state(self, file_path=None, file_name="test"):
        self.config.object_serializer.set_object(object=self)
        self.config.object_serializer.safe_object(file_path=file_path,file_name=file_name)
    
    @classmethod
    def load_controller_state(cls,config:Blueprint_main_controller, file_path=None, file_name="test") -> "Cg_flex_controller":
        serializer = config.object_serializer
        serializer.load_object(file_path=file_path,file_name=file_name)
        object = serializer.return_object()
        return object

    def safe_nodelists_dependencies_to_dict(self, file_path=None): # is saving the nodelist within the dependency_controller including its nested dependencyobject
        counter = 1
        for dependency_controller in self.dependency_controller:
            dependency_controller.safe_nodelist_with_dependencies_pikle_gpy_models_to_dict(file_name=f"default_dependencysafe_graphcomponent_nr{counter}", file_path=file_path)
            counter += 1

    def load_nodelists_dependencies_from_dict(self, file_path=None):# loading the nodelist within the dependency_controller including its nested dependencyobject
        counter = 1
        for dependency_controller in self.dependency_controller:
            dependency_controller.load_nodelist_with_dependencies_pikle_gpy_models_from_dict(file_name=f"default_dependencysafe_graphcomponent_nr{counter}", file_path=file_path)
            counter += 1

    def safe_nodelists_graph(self, file_path=None): # lets user to safe nodelists for graphs individually
        counter = 0
        for graph in self.graph_controller:
            graph.safe_nodelist_pikle(file_name=f"default_grahphsafe_graphcomponent_nr{counter}", file_path=file_path)
            counter += 1
    
    def load_nodelists_graph(self, file_path=None): # lets user load nodelists for graphs individually
        counter = 0
        for graph in self.graph_controller:
            graph.load_nodelist_pikle(file_name=f"default_grahphsafe_graphcomponent_nr{counter}",file_path=file_path)
            counter += 1

  