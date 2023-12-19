import cgflex_project.Shared_Classes.distributions as distributions
from cgflex_project.Module_Graphmaker.sourcemaker import ISource_setter
from cgflex_project.Module_Graphmaker.layermaker import ILayersetter
from cgflex_project.Module_Graphmaker.sinkmaker import ISink_setter
from cgflex_project.Module_Graphmaker.verticemaker import IEdgemaker
from cgflex_project.Module_Graphmaker.nodemaker import INodemaker
from typing import Any, List, Type, Tuple, Optional
from dataclasses import dataclass, field
from  cgflex_project.Module_Dependencymaker._dependencymaker_initial_value_distributions import IInitial_value_distribution_collection
from  cgflex_project.Module_Dependencymaker._dependencymaker import IDependency_setter
from  cgflex_project.Module_Dependencymaker.sampling_tsd import ITsd_strategy_setter
from  cgflex_project.Module_Dependencymaker._functionmaker_extreme_values import INormalizer, Normalizer_minmax_stretch
#from Controllermodule_extensions import IObject_serializer, IController_coordinator
from cgflex_project.Module_Graphmaker.Graphmodule_extensions import IGraph_processor
from cgflex_project.Module_Sampler.Samplingmodule_extensions import IData_exporter
#from Samplingmodule import Sampling_controller
#from cgflex_project.Controllermodule_extensions import IObject_serializer

from  cgflex_project.Module_Dependencymaker._dependencymaker_tsd_functions import ITsd_function_collection


@dataclass
class Blueprint_graph():

    nodemaker: INodemaker
    layermaker: ILayersetter
    sourcemaker: ISource_setter
    sinkmaker: ISink_setter
    edgemaker: IEdgemaker
    graphprocessor: IGraph_processor
    



@dataclass
class Blueprint_dependency:
    dependency_setter : IDependency_setter
    initial_value_distributions : IInitial_value_distribution_collection
    range_of_output : Tuple[float, float]
    tsd_collection :  ITsd_function_collection

    def __post_init__(self):
        # set the range after initialization
        self.tsd_collection.set_range(range=self.range_of_output)

@dataclass
class Blueprint_sampling:
    data_exporter:IData_exporter


#@dataclass
#class Blueprint_main_controler:
  #  object_serializer : IObject_serializer
  #  controller_coordinator : IController_coordinator
 #   sampling_controller : Sampling_controller
    




