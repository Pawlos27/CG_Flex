
from dataclasses import dataclass, field
import cgflex_project.Module_Dependencymaker._kernelcombinator as _kernelcombinator 
import cgflex_project.Module_Dependencymaker._kernel_collection as _kernel_collection
import cgflex_project.Module_Dependencymaker._errortermmaker as _errortermmaker
import cgflex_project.Module_Dependencymaker._functionmaker as _functionmaker
import cgflex_project.Module_Dependencymaker._errortermmaker_collection as _errortermmaker_collection
import cgflex_project.Module_Dependencymaker._functionmaker_extreme_values as _functionmaker_extreme_values
import cgflex_project.Module_Dependencymaker._dependencymaker_initial_value_distributions as _dependencymaker_initial_value_distributions
import cgflex_project.Module_Dependencymaker._dependencymaker as _dependencymaker
from cgflex_project.Shared_Classes.blueprints import Blueprint_dependency, Blueprint_graph, Blueprint_sampling
from cgflex_project.Shared_Classes.blueprint_main_controller import Blueprint_main_controller
from dataclasses import dataclass, field
import cgflex_project.Shared_Classes.distributions as distributions
import cgflex_project.Module_Graphmaker.sourcemaker as sourcemaker 
import cgflex_project.Module_Graphmaker.layermaker as layermaker
import cgflex_project.Module_Graphmaker.sinkmaker as sinkmaker
import cgflex_project.Module_Graphmaker.verticemaker as verticemaker
import cgflex_project.Module_Graphmaker.nodemaker as nodemaker 
from dataclasses import dataclass, field
from  cgflex_project.Module_Dependencymaker._dependencymaker_initial_value_distributions import IInitial_value_distribution_collection
from  cgflex_project.Module_Dependencymaker._dependencymaker import IDependency_setter

import cgflex_project.Main_Controller.Controllermodule_extensions as Controllermodule_extensions
import cgflex_project.Module_Graphmaker.Graphmodule_extensions as Graphmodule_extensions
import cgflex_project.Module_Sampler.Samplingmodule_extensions as Samplingmodule_extensions
from cgflex_project.Module_Sampler.Samplingmodule import Sampling_controller
import cgflex_project.Module_Dependencymaker._inputloader as _inputloader
import cgflex_project.Module_Dependencymaker._dependencymaker_tsd_functions as _dependencymaker_tsd_functions


#Graphcontroller configurations


blueprint_test=Blueprint_graph( nodemaker= nodemaker.Nodemaker(number_of_nodes=50, number_of_dimensions_thorus=1 ,scale_per_n=0.1, 
                                                                l_distribution=nodemaker.Nodemaker_distribution_uniform(), t_distribution=nodemaker.Nodemaker_distribution_uniform()),
                               layermaker= layermaker.Layer_setter_equinumber(10),
                               sourcemaker= sourcemaker.Source_setter_by_probability (number_of_sources=2, shift_parameter=2),
                               sinkmaker= sinkmaker.Sink_setter_by_probability(number_of_sinks=2, shift_parameter=2),
                               edgemaker= verticemaker.edgemaker(layerlimit=2, number_of_edges=60, edgeprobability_layer= verticemaker.Edge_probability_by_distance_decreasing_inverse(), 
                                                                 edgeprobability_thorus= verticemaker.Edge_probability_by_distance_decreasing_exponentially(), edge_outdegree_max= 10, edge_indegree_max= 30 ),
                               graphprocessor= Graphmodule_extensions.Graph_processor_networx_solo())

config_graph_test=Blueprint_graph( nodemaker= nodemaker.Nodemaker(number_of_nodes=40, number_of_dimensions_thorus=1 ,scale_per_n=0.1,  l_distribution=nodemaker.Nodemaker_distribution_uniform(),
                                                                   t_distribution=nodemaker.Nodemaker_distribution_uniform()),
                               layermaker= layermaker.Layer_setter_equinumber(15),
                               sourcemaker= sourcemaker.Source_setter_by_probability (number_of_sources=4, shift_parameter=2),
                               sinkmaker= sinkmaker.Sink_setter_by_probability(number_of_sinks=4, shift_parameter=2),
                               edgemaker= verticemaker.edgemaker(layerlimit=15, number_of_edges=55, edgeprobability_layer= verticemaker.Edge_probability_by_distance_decreasing_inverse(exponential_factor=1),
                                                                  edgeprobability_thorus= verticemaker.Edge_probability_by_distance_decreasing_exponentially(exponential_factor=1), edge_outdegree_max= 4, edge_indegree_max= 5 ),
                               graphprocessor= Graphmodule_extensions.Graph_processor_networx_solo())

config_graph_random =Blueprint_graph( nodemaker= nodemaker.Nodemaker(number_of_nodes=40, number_of_dimensions_thorus=1 ,scale_per_n=0.1,  l_distribution=nodemaker.Nodemaker_distribution_uniform(),
                                                                   t_distribution=nodemaker.Nodemaker_distribution_uniform()),
                               layermaker= layermaker.Layer_setter_continuous(),
                               sourcemaker= sourcemaker.Source_setter_by_probability (number_of_sources=4, shift_parameter=2),
                               sinkmaker= sinkmaker.Sink_setter_by_probability(number_of_sinks=4, shift_parameter=2),
                               edgemaker= verticemaker.edgemaker(layerlimit=15, number_of_edges=50, edgeprobability_layer= verticemaker.Edge_probability_by_distance_decreasing_exponentially(exponential_factor=0.001),
                                                                  edgeprobability_thorus= verticemaker.Edge_probability_by_distance_decreasing_inverse(exponential_factor=0.01), edge_outdegree_max= 10, edge_indegree_max= 10 ),
                               graphprocessor= Graphmodule_extensions.Graph_processor_networx_solo())






blueprint_graph_linear_test=Blueprint_graph( nodemaker= nodemaker.Nodemaker(number_of_nodes=55, number_of_dimensions_thorus=1 ,scale_per_n=0.1,  l_distribution=nodemaker.Nodemaker_distribution_uniform(), t_distribution=nodemaker.Nodemaker_distribution_uniform()),
                               layermaker= layermaker.Layer_setter_equispace(20),
                               sourcemaker= sourcemaker.Source_setter_by_probability (number_of_sources=4, shift_parameter=2),
                               sinkmaker= sinkmaker.Sink_setter_by_probability(number_of_sinks=4, shift_parameter=2),
                               edgemaker= verticemaker.edgemaker(layerlimit=8, number_of_edges=45, edgeprobability_layer= verticemaker.Edge_probability_by_distance_decreasing_inverse(), edgeprobability_thorus= verticemaker.Edge_probability_by_distance_decreasing_exponentially(exponential_factor=5), edge_outdegree_max= 3, edge_indegree_max= 5 ),
                               graphprocessor= Graphmodule_extensions.Graph_processor_networx_solo())

# graph different configs for tests, andere verteilungen, handgemachte sources und sinks


#Dependencycontroller configurations

blueprint_dependency_test_3_different_range = Blueprint_dependency( dependency_setter=_dependencymaker.Dependency_setter_default( kernel_combination_maker= _kernelcombinator.Kernelcombinator_random_picking( kernel_operator_collection=_kernel_collection.Kernel__operator_collection_default(),
                                                                                                                                                                                                               kernel_selector= _kernelcombinator.Kernel_selector_random(max_dimensions_per_kernel= 1,
                                                                                                                                                                                                                                                                         kernel_collection= _kernel_collection.Kernel_collection_general_full())),
                                                 errorterm_maker= _errortermmaker.Errorterm_maker_default(errorterm_collection=_errortermmaker_collection.Error_term_collection_solo_normal(),maximum_tolerance=0.1),
                                                 function_maker=_functionmaker.Function_maker_evenly_discontinuity_in_one_dimension(inputloader= _inputloader. Inputloader_for_solo_random_values(),discontinuity_frequency= 0.2,  maximum_discontinuities=2, discontinuity_reappearance_frequency=0.3, extreme_value_setter=_functionmaker_extreme_values.Extreme_value_setter_solo_dimensionmax(resolution=100),normalizer= _functionmaker_extreme_values.Normalizer_minmax_stretch())),
                                                 initial_value_distributions= _dependencymaker_initial_value_distributions.Initial_value_distribution_random_full(),
                                                 range_of_output=(0,1),
                                                 tsd_collection= _dependencymaker_tsd_functions.Tsd_function_collection_full())


#Samplingcontroller configurations
blueprint_sampling= Blueprint_sampling(data_exporter=Samplingmodule_extensions.Csv_exporter_for_id_value_pairs())

#Maincontroller configurations

blueprint_controller_test = Blueprint_main_controller(object_serializer=Controllermodule_extensions.Object_serializer_pickle(), controller_coordinator= Controllermodule_extensions.Controller_coordinator_exact_order(list_of_dependency_configs=[blueprint_dependency_test_3_different_range],list_of_graph_configs=[blueprint_test]), sampling_controller=Sampling_controller(config=blueprint_sampling))
