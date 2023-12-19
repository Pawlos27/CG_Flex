
from typing import Any, List, Type, Tuple, Optional
import pickle
import os
import random
from abc import ABCMeta, abstractstaticmethod, abstractmethod
from  cgflex_project.Module_Dependencymaker.Dependencymodule import Dependency_controller
from cgflex_project.Module_Graphmaker.Graphmodule import Graph_controller
from cgflex_project.Shared_Classes.blueprints import Blueprint_graph, Blueprint_dependency, Blueprint_sampling


class IObject_serializer(metaclass=ABCMeta):
    @abstractmethod
    def safe_object(self, file_path:Optional[str], file_name="test"):
     """Interface Method"""
    @abstractmethod
    def load_object(self,file_path:Optional[str], file_name="test"):
     """Interface Method"""
    @abstractmethod
    def set_object(self, object):
     """Interface Method"""
    @abstractmethod
    def return_object(self):
     """Interface Method"""


class Object_serializer_pickle(IObject_serializer):
    def __init__(self):
        self.object = None

    def set_object(self, object):
        self.object = object
        
    def return_object(self):
        return self.object
    
    def safe_object(self, file_path:Optional[str], file_name="test"):
        if file_path == None:
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)
        print(f"safed path {file_path}")
        with open(file_path, 'wb') as file:
            pickle.dump(self.object, file)

    def load_object(self,file_path:Optional[str], file_name="test"):
        if file_path == None:
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)
        print(f"loaded path object{file_path}")
        with open(file_path, 'rb') as file:
            self.object = pickle.load(file)



class IController_coordinator(metaclass=ABCMeta):
    
    def make_list_graph_controller(self, size:int) ->List[Type[Graph_controller]]:
     """Interface Method"""
    def make_list_dependency_controller(self, size:int)->  List[Type[Dependency_controller]]:
     """Interface Method"""


class Controller_coordinator_random(IController_coordinator):

    def __init__(self, list_of_graph_configs: List[Blueprint_graph], list_of_dependency_configs:List[Blueprint_dependency]):
        self.list_of_graph_configs = list_of_graph_configs
        self.list_of_dependency_configs = list_of_dependency_configs

    def make_list_graph_controller(self, size:int):
        list_controller = []
        for i in range(size):
            selected_config = random.choice(self.list_of_graph_configs)
            single_controller = Graph_controller(config=selected_config)
            list_controller.append(single_controller)
        print(list_controller)
        return list_controller

    def make_list_dependency_controller(self, size:int):
        list_controller = []
        for i in range(size):
            selected_config = random.choice(self.list_of_dependency_configs)
            single_controller = Dependency_controller(config=selected_config)
            list_controller.append(single_controller)
        return list_controller
    
    

class Controller_coordinator_exact_order(IController_coordinator):

    def __init__(self, list_of_graph_configs: List[Blueprint_graph], list_of_dependency_configs:List[Blueprint_dependency]):
        self.list_of_graph_configs = list_of_graph_configs
        self.list_of_dependency_configs = list_of_dependency_configs

    def make_list_graph_controller(self, size:int):
        list_controller = []
        for i in range(size):
           index = i % len(self.list_of_graph_configs)
           selected_config = self.list_of_graph_configs[index]
           single_controller = Graph_controller(config=selected_config)
           list_controller.append(single_controller)
        return list_controller

    def make_list_dependency_controller(self, size:int):
        list_controller = []
        for i in range(size):
           index = i % len(self.list_of_dependency_configs)
           selected_config = self.list_of_dependency_configs[index]
           single_controller = Dependency_controller(config=selected_config)
           list_controller.append(single_controller)
           
        return list_controller
