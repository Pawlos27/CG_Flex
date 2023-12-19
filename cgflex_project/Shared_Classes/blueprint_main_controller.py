
from dataclasses import dataclass, field
from cgflex_project.Main_Controller.Controllermodule_extensions import IObject_serializer, IController_coordinator
from cgflex_project.Module_Graphmaker.Graphmodule_extensions import IGraph_processor
from cgflex_project.Module_Sampler.Samplingmodule_extensions import IData_exporter
from cgflex_project.Module_Sampler.Samplingmodule import Sampling_controller



@dataclass
class Blueprint_main_controller:
    object_serializer : IObject_serializer
    controller_coordinator : IController_coordinator
    sampling_controller : Sampling_controller
    




