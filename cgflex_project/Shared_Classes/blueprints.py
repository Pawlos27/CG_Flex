import cgflex_project.Shared_Classes.distributions as distributions
from cgflex_project.Module_Graphmaker.sourcemaker import ISource_setter
from cgflex_project.Module_Graphmaker.layermaker import ILayersetter
from cgflex_project.Module_Graphmaker.sinkmaker import ISink_setter
from cgflex_project.Module_Graphmaker.verticemaker import IEdgemaker
from cgflex_project.Module_Graphmaker.nodemaker import INodemaker
from typing import Any, List, Type, Tuple, Optional
from dataclasses import dataclass, field
from  cgflex_project.Module_Dependencymaker._dependencymaker_initial_value_distributions import IInitial_value_distribution_collection, Initial_value_distribution_random_full
from  cgflex_project.Module_Dependencymaker._dependencymaker import IDependency_setter
from  cgflex_project.Module_Dependencymaker._functionmaker_extreme_values import INormalizer, Normalizer_minmax_stretch
from cgflex_project.Module_Graphmaker.Graphmodule_extensions import IGraph_processor, Graph_processor_networx_solo
from cgflex_project.Module_Sampler.Samplingmodule_extensions import IData_exporter, Csv_exporter_for_id_value_pairs
#from cgflex_project.Controllermodule_extensions import IObject_serializer
from  cgflex_project.Module_Dependencymaker._dependencymaker_tsd_functions import ITsd_function_collection, Tsd_function_collection_full


@dataclass
class Blueprint_graph():
    """A dataclass representing the blueprint for graph configuration objects.
    It is used for the Graph_controller.This class defines the components necessary for constructing, manipulating and analysing a graph. 
    It includes instances of sub classes providing the functionality.

    """

    nodemaker: INodemaker
    """Component for determining the number and distribution of nodes."""
    layermaker: ILayersetter
    """Component for setting the type and number of layers ."""
    sourcemaker: ISource_setter
    """Component for setting sources."""
    sinkmaker: ISink_setter
    """Component for setting sinks."""
    edgemaker: IEdgemaker
    """Component for generating edges through various constraints."""
    graphprocessor: IGraph_processor = Graph_processor_networx_solo()
    """Component for graph representation and analysis functionalities."""
    



@dataclass
class Blueprint_dependency:
    """ A dataclass representing the blueprint for dependency configuration objects.
    It is used for the Dependency_controller. This class sets parameters for the generation of dependencies in the graph, including 
    dependency setters, value distributions, and a range of output values. The configuration 
    is detailed, covering aspects from kernel combination to error term creation and normalization."""

    dependency_setter : IDependency_setter
    """Component for setting dependencies(Instances of this class contain a nesting of many classes, ranging from kernel combination to function generation and error term generation)."""
    initial_value_distributions : IInitial_value_distribution_collection = Initial_value_distribution_random_full()
    """Pre-selected value distributions that can be assigned to nodes."""
    range_of_output : Tuple[float, float] = (0,1)
    """The target value range ."""
    tsd_collection :  ITsd_function_collection = Tsd_function_collection_full()
    """A collection of TSD functions used as substitutes for dependencies or value distributions."""

    def __post_init__(self):
        """Sets the parameters of random value distributions and tsd functions regarding to the set range of output."""
        # set the range after initialization
        self.tsd_collection.set_range(range=self.range_of_output)
        self.initial_value_distributions.initialize_distributions(range=self.range_of_output)


@dataclass
class Blueprint_sampling:
    """ A dataclass representing the blueprint for the sampling controller configuration object. 
    This configuration is simpler compared to others, mainly containing the data_exporter component. 
    The sampling controller's functionality is mostly realized through its own methods.
    """
    data_exporter:IData_exporter = Csv_exporter_for_id_value_pairs()
    """Component responsible for exporting data during sampling."""






