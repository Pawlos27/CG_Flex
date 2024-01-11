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
    def __init__(self, config: Blueprint_dependency):
        self.inputs_dependency = config
        self.nodelist = None
        self.list_of_sources = None

    def load_nodelist_graph(self, nodelist: List[Type[Nodeobject]]): 
        self.nodelist = nodelist
        
    def make_dependencies(self):    # sources are set as fixed values by default
        self.nodelist.sort()    
        self._set_list_of_sources() 
        self.set_sources_to_fixed_values()
        for node in self.nodelist:
            if len(node.parents) >0 :
                node.dependency = self.inputs_dependency.dependency_setter.set_dependencies(node=node, range_of_output= self.inputs_dependency.range_of_output)
            else:
                pass

    def reset_config(self,inputs_dependency: Blueprint_dependency):
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
        if self.list_of_sources == None:
            raise ValueError("no nodelist! load nodelist into dependencycontroller before trying to access a list of sources")
        else: 
            return self.list_of_sources

    def reset_dependencies_specific(self,node_ids: List[int]):
        for node_id in node_ids:
            self.nodelist[node_id].dependency = self.inputs_dependency.dependency_setter.set_dependencies(node=self.nodelist[node_id])

    def replace_dependencies_by_initial_distributions(self,node_ids: List[int]):
        for node_id in node_ids:
            self.nodelist[node_id].dependency = self.inputs_dependency.initial_value_distributions.get_distribution()

    def replace_all_dependencies_by_initial_distributions(self):
        for node in self.nodelist:
            node.dependency = self.inputs_dependency.initial_value_distributions.get_distribution()

    def replace_dependencies_by_single_values(self,node_ids_and_values: List[Tuple[int, float]]):
        for tuple in node_ids_and_values:
            if tuple[1]>1 or tuple[1]< 0:
                raise ValueError("values need to be between 0 and 1")
            else:
             self.nodelist[tuple[0]].dependency = tuple[1]
    
    def replace_dependencies_by_single_values_from_random_distribution(self,node_ids: List[int]):
        distribution = self.inputs_dependency.initial_value_distributions.get_distribution()
        for node_id in node_ids:
            self.nodelist[node_id].dependency = distribution.get_value_from_distribution()
    
    def replace_dependencies_by_tsd_function(self,node_ids: List[int]):
        for node_id in node_ids:
            tsd_function = self.inputs_dependency.tsd_collection.get_tsd_function()
            self.nodelist[node_id].dependency = tsd_function

    def set_sources_as_tsd_function(self):
        for source in self.list_of_sources:
            tsd_function = self.inputs_dependency.tsd_collection.get_tsd_function()
            self.nodelist[source].dependency = tsd_function

    def set_sources_as_distributions(self):
        for source in self.list_of_sources:
            self.nodelist[source].dependency = self.inputs_dependency.initial_value_distributions.get_distribution()

    def set_sources_to_fixed_values(self):
        distribution = self.inputs_dependency.initial_value_distributions.get_distribution()
        for source in self.list_of_sources:
            self.nodelist[source].dependency = distribution.get_value_from_distribution()

    def get_nodelist_with_dependencies(self):
        if self.nodelist == None:
            raise ValueError("there is no graph yet, please make a graph and add dependencies")
        elif self.nodelist[0].dependency == None:
            raise ValueError("there are no dependencies in the nodelist yet, please add dependencies")
        else:
            return self.nodelist

    def show_dependencies(self,ids:List[int],resolution:int): 
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, (float, int)):
                        print(f" Dependency at Node_ID: {id} is a Value :  {node.dependency} ")
                    elif isinstance(node.dependency, IDistributions):
                        print(f" Dependency at Node_ID: {id} is a Distribution")
                        node.dependency.plot_distribution(label=f"distribution for node-id {id}")
                    elif isinstance(node.dependency, Dependencies):
                        print(f" Dependency at Node_ID: {id} is a Dependency_Object")
                        node.dependency.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=resolution, label=f" node_id ={id}   Function")
                        node.dependency.function_model.show_function_borders()
                        node.dependency.errorterm_model.errorterm_distribution.show_error_distribution(label =f" node_id ={id}   ")
                    elif isinstance(node.dependency, ITsd_functions):
                        node.dependency.plot_function(label=f"tsd function for node-id ={id}")
                    else:
                        print("dependency is empty yet, please make dependencies")

    def show_dependencies_enforce_3d_plot(self,ids:List[int],resolution:int): 
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, (float, int)):
                        print(f" Dependency at Node_ID: {id} is a Value :  {node.dependency} ")
                    elif isinstance(node.dependency, IDistributions):
                        print(f" Dependency at Node_ID: {id} is a Distribution")
                        node.dependency.plot_distribution()
                    elif isinstance(node.dependency, Dependencies):
                        print(f" Dependency at Node_ID: {id} is a Dependency_Object")
                        node.dependency.function_model.show_functions_3d_plot_for_first_two_dimensions_when_possible(resolution=resolution, label=f" node_id ={id}   Function")
                        node.dependency.function_model.show_function_borders()
                        node.dependency.errorterm_model.errorterm_distribution.show_error_distribution(label =f" node_id ={id}   ")
                    else:
                        print("dependency is empty yet, please make dependencies")

    def show_dependency_function(self,ids:List[int],resolution:int): 
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=resolution)
                    else:
                        print("no functions in dependency for {id}")

    def show_dependency_function_enforce_3d_plot(self,ids:List[int],resolution:int): 
        for id in ids:
            for node in self.nodelist:
                if node.id == id:
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_for_first_two_dimensions_when_possible(resolution=resolution)
                    else:
                        print("no functions in dependency for {id}")
    def show_dependency_errorterm(self,ids:List[int],resolution:int): 
        for id in ids:    
            for node in self.nodelist:
                if node.id == id:                  
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_if_exactly_two_dimensions(resolution=resolution)
                    else:
                        print("no errorterm in dependency")

    def show_dependency_errorterm_enforce_3d_plot(self,ids:List[int],resolution =30): 
        for id in ids:    
            for node in self.nodelist:
                if node.id == id:                  
                    if isinstance(node.dependency, Dependencies):
                        node.dependency.function_model.show_functions_3d_plot_for_first_two_dimensions_when_possible(resolution=resolution)
                    else:
                        print("no errorterm in dependency")

    def print_kernels_used(self,ids:List[int]): 
        pass

    def safe_nodelist_with_dependencies_pikle_gpy_models_to_dict(self,file_name:str, file_path:Optional[str]):
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
        if file_path == None:
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'rb') as file:
            dicted_nodelist = pickle.load(file)
        
        self.nodelist = self._make_nodelist_with_Gpy_from_dict(nodelist=dicted_nodelist)

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
                if isinstance(node.dependency.errorterm_model.errorterm_distribution, Error_distribution_normal_variable_variance):
                    print(f"durchlauf {counter}  errorterm davor" )
                    errorterm_functionlist = node.dependency.errorterm_model.errorterm_distribution.function_model.functions
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
            
                if isinstance(node.dependency.errorterm_model.errorterm_distribution, Error_distribution_normal_variable_variance):
                    errorterm_functionlist = node.dependency.errorterm_model.errorterm_distribution.function_model.functions
                    for function in errorterm_functionlist:
                        undicted_model = GPy.models.GPRegression.from_dict(function.function_model)
                        function.function_model = undicted_model
        return undicted_nodelist
        



if __name__ == "__main__":
    pass
 