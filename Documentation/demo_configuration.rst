Demo_Configuration
#########################

Table of Contents
-----------------

1. `Config_Graph`_
2. `Config_Dependency`_
3. `Config_Sampler`_
4. `Config_Main`_

Initially, the setup must be carried out. All configuration objects for the sub-controllers need to be parameterized and instantiated. Below we provide a brief overview of the configuration objects and what parameters they control.

.. _Config_Graph:

Config_Graph
------------

The most important settings in the Graphmaker are summarized as follows:

- **NodeMaker**: Determines the number of nodes and their distribution.
- **SourceMaker**: Determines the number of sources and the type of their distribution.
- **SinkMaker**: Determines the number of sinks and the type of their distribution.
- **LayerMaker**: Determines the type of layer distribution and their quantity.
- **EdgeMaker**: Sets the number of edges, functions for probability calculation, layer limits, as well as restrictions on the in- and out-degree.
- **GraphProcessor**: Stores functionalities for the representation and analysis of the graph.

Sensitive Parameters in Config Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Settings in the config for the graph are particularly sensitive due to our methodology of node and edge generation, and can have significant impacts on the structure of the graph:

1. **Layer distributions and distance functions**: 
   Inaccurate settings of these parameters can complicate partner finding. Restrictions within the layers and across multiple layers reduce the number of potential partners. Additionally, potential partners have their own restrictions regarding their connections to other nodes.

2. **Problem of not finding partners**: 
   This can lead to a significant increase in the number of sinks and sources, or even to isolated nodes in the graph.

3. **Weight Functions and layer size**: 
   The combination of very thick layers and the usage of "Linear decreasing weight Functions" which are static and do not dynamically adapt to the space(means even if the space grows because we use more nodes, its threshold and steepness stays the same)  .
   "Linear decreasing weight Functions" can cause that distances outside a threshold result in weights=0 , because of the formula how probabilities for connections are calculated this reduces also the probability of a connection to zero. So if a node has only nodes within the threshold radius, that are within the same layer, the forming of a connection is possible under such circumstances.

4. **Combination of layer restrictions, layer jumps, and node degrees**: 
   The number and type of layers, combined with the limitation of their jumps and the node degree restrictions, can lead to no more free partners being available. This results in an increase in sinks and sources.

.. _Config_Dependency:

Config_Dependency
-----------------

The configuration for Dependencies encompasses a range of parameters essential for generating Dependencies. Given the nested nature of the objects in this class, a breakdown of the configuration is provided for clarity:

- **Dependency_setter**
  
  - **Kernel_combination_maker**: Responsible for creating combined kernels.

    - **Kernel_selector**: Methodology for selecting kernels.

  - **Errorterm_maker**: Facilitates the creation of Error Terms.

    - **Errorterm_collection**: A pre-selection process for determining complexity.
    - **Maximum_tolerance**: A parameter that sets the allowable range for Error Terms.

  - **Function_maker**

    - **Inputloader**: Determines the type of training data for the model.
    - **Discontinuity_frequency**: Parameter controlling the frequency of discontinuities.
    - **Maximum_discontinuities**: Limit on the number of discontinuities.
    - **Discontinuity_reappearance_frequency**: Governs the frequency of clustered discontinuities..
    - **Normalizer**: Methodology for normalizing values.

        - **Extreme_value_setter**: Strategy for defining function values within the targeted range.

- **Initial_value_distributions**: Pre-selects distributions that can be assigned to nodes in place of Dependencies.

- **Range_of_output**: Defines the target value range for discontinuities.

- **Tsd_collection**: Offers a selection of TSD (Time Series Data) functions as alternatives to Dependencies.

.. _Config_Sampler:

Config_Sampler
--------------

The configuration of the Sampling-Controller is comparatively less complex since most of its functionality is realized through its methods. Here, the configuration only contains one object, namely the data_exporter.

.. _Config_Main:

Config_Main
-----------

The configuration of the Main-Controller includes all settings for the Main-Controller but also integrates the blueprints of the previous controllers:

- **object_serializer**: Provides storage and loading functionalities.

- **controller_coordinator**: Responsible for instantiating the sub-controllers within the Main-Controller.
 
  - **Blueprint_dependency**: Configuration objects for dependency control.
  - **Blueprint_graph**: Configuration objects for graph control.
-
 **sampling_controller**: Manages the sampling process.
  
  - **Blueprint_sampling**: Configuration object for the sampling process.
