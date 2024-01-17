
from dataclasses import dataclass, field
from cgflex_project.Main_Controller.Controllermodule_extensions import IObject_serializer, IController_coordinator, Object_serializer_pickle
from cgflex_project.Module_Graphmaker.Graphmodule_extensions import IGraph_processor
from cgflex_project.Module_Sampler.Samplingmodule_extensions import IData_exporter
from cgflex_project.Module_Sampler.Samplingmodule import Sampling_controller



@dataclass
class Blueprint_main_controller:
    """ A dataclass representing the blueprint for the main controller configuration object.
    Objects of this class serve as the configuration objects  for the main controller.
    """

    controller_coordinator : IController_coordinator
    """An instance of IController_coordinator, it coordinates the instantaion of the Dependency_controller and the Graph_controller, by passing its configuration objects to it.
    """  
    sampling_controller : Sampling_controller 
    """An instance of Sampling_controller for handling sampling processes, also instantiated with its config object.
    """
    object_serializer : IObject_serializer = Object_serializer_pickle()
    """ An instance of IObject_serializer for object serialization tasks.
    """
    




