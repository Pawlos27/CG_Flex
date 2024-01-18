
from typing import Any, List, Type, Tuple, Optional
import pickle
import os
import random
from abc import ABCMeta, abstractstaticmethod, abstractmethod
from  cgflex_project.Module_Dependencymaker.Dependencymodule import Dependency_controller
from cgflex_project.Module_Graphmaker.Graphmodule import Graph_controller
from cgflex_project.Shared_Classes.blueprints import Blueprint_graph, Blueprint_dependency, Blueprint_sampling


class IObject_serializer(metaclass=ABCMeta):
    """
    Interface for object serialization.

    This interface defines methods for saving and loading objects, typically used for data persistence.
    """
    @abstractmethod
    def safe_object(self, file_path:Optional[str], file_name="test"):
     """
        Saves the object to a specified file path.

        Args:
            file_path (Optional[str]): The file path where the object is to be saved. If None, a default path is used.
            file_name (str, optional): The name of the file. Defaults to "test".
        """
    @abstractmethod
    def load_object(self,file_path:Optional[str], file_name="test"):
     """
        Loads an object from a specified file path.

        Args:
            file_path (Optional[str]): The file path from which the object is to be loaded. If None, a default path is used.
            file_name (str, optional): The name of the file. Defaults to "test".
        """
    @abstractmethod
    def set_object(self, object):
     """
        Sets the object to be serialized.

        Args:
            object: The object to be set for serialization.
        """
    @abstractmethod
    def return_object(self):
     """
        Returns the currently set object.

        Returns:
            The object that was set for serialization.
        """


class Object_serializer_pickle(IObject_serializer):
    """
    Implementation of IObject_serializer using Python's pickle module.

    This class provides methods to serialize and deserialize objects using pickle.
    """
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
    """
    Abstract base class defining the interface for controller coordinators which are creating instantaions of the Graph_controller and the Dependency_controller to be used in the Main Controller
      The Idea is that we want to use a variable number of graph components and we would like for each component to be able to use different configuration objects to be different in size and properties.
    """
    
    def make_list_graph_controller(self, size:int) ->List[Type[Graph_controller]]:
     """
        Creates a list of graph controller instances.

        Args:
            size (int): The number of graph controllers to create.

        Returns:
            List[Type[Graph_controller]]: A list of graph controller instances.
        """
    def make_list_dependency_controller(self, size:int)->  List[Type[Dependency_controller]]:
     """
        Creates a list of dependency controller instances.

        Args:
            size (int): The number of dependency controllers to create.

        Returns:
            List[Type[Dependency_controller]]: A list of dependency controller instances.
        """


class Controller_coordinator_random(IController_coordinator):
    """
    Implementation of IController_coordinator that randomly selects configurations 
    to create graph and dependency controller instances.
    """

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
    """
    Implementation of IController_coordinator that creates controller instances in the exact order of given configurations.
    """

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
