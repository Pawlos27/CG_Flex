# [CG_Flex Readme]


## Table of Contents
1. [Description](#Description)
2. [Installation](#Installation)
3. [Documentation](#Documentation)
4. [Feautures](#Feautures)
5. [Usage](#Usage)
6. [Scripts](#Scripts)
7. [Demos](#Demos)
8. [Extensions](#Extensions)
9. [Remaining_Tasks](#Remaining_Tasks)


## **Description**

This framework is designed to generate synthetic data for evaluating causal discovery algorithms. It addresses the challenge of unrealistic synthetic data by allowing for the controlled generation of datasets with known dependencies. This is crucial in fields where understanding causal relationships is vital, and real observational data often lack clear causal structures.


## **Key Features**

- **Graph Generation**: Facilitates the generation of Directed Acyclic Graphs (DAGs). Enables users to influence graph properties like number of nodes, edges,layers, and maximum indegree or outdegree. Also choosing methodologies for defining these properties.

- **Dependency Definition**: Each node's value in the DAG is represented by a dependency function D(x) = F(x) + E(x), where x denotes the values of its parent nodes. F(x) is a deterministic function based on these values, and E(x) represents an error term, introducing a deviation variable. 
F(x): The framework supports the creation of random functions using Gaussian processes for both linear and nonlinear functions.
E(x): It extends these functions with error terms like normal, mixture, and conditional distributions.

- **Interactive Sampling**: Enables users to adjust dependencies or replace node dependencies with fixed values between sampling rounds. This enhances network analysis and provides valuable insights for algorithm interaction.

### Advantages

- Provides a baseline for simulating realistic scenarios.
- Enables a gradual increase in complexity of the dependencies("linear function < +nonlinear < + normal distribution < + mixture distribution< + consitional distribution). This allows challenging existing analytical methods.
- Offers interactive model manipulation for deeper understanding of variable interactions.
- Ensures known dependencies("Ground Truth") for direct comparison in algorithm evaluations, improving algorithm assessment.

This framework is essential for improving causal discovery algorithms, providing the flexibility and control needed for effective evaluation and comparison under diverse conditions.



## **Framework Structure**

Our Framework consists of a Main Controller and three Sub Controllers for graph, dependency, and sampling control. The Main Controller, serving as the primary user interface, coordinates core functionalities with the Sub Controllers. Configuration objects are crucial for initializing these controllers, allowing users to choose specific class implementations and parameters. This setup, along with the ability for users to introduce custom class implementations, ensures adaptability across various data generation scenarios.


## **Installation**

The Framework uses poetry to manage the projects dependencies and requires python 3.9.6 to 3.9.16 for the dependencies to work. The python version can be created via pyenv.

install pyenv

install python version:

```bash
pyenv install 3.9.6
```

install poetry

install project dependencies:

```bash
poetry install
```


## **Documentation**
The project includes a detailed documentation that provides explanations for all modules, classes, and methods, as well as the significance and impact of each parameter. The documentation also features UML class and sequence diagrams. Additionaly it gives further information on how to use the scripts.

The htmls are in the [Documentation](./Documentation/_build/html/) directory.

Otherwise you can also generate your own htmls.
**Generate Documentation**: 
The documentation is generated from the docstrings of the code using Sphinx, which is installed alongside Poetry. All necessary files and settings are in the Documentation folder.

After navigating to the Documentation directory you can generate html files for the documentation:

simply run
```bash
make html
```

when using powershell you might need to use 

```bash
.\make.bat html
```

## **Usage**


The steps of the Usage of the framework:

1. **Preparation**:
Begin with the initial setup. Create configuration objects for the Sub Controllers and then for the Main Controller. This involves parameterization and instantiation of these components.
[Script for configuration setup](scripts/config_objects.py) 

2. **Initiation**:
Instantiate the Main Controller using the previously created configuration object as a parameter.
[Script for using the Framework simple](scripts/usage_ready_script_simple.py) 
3. **Model Generation**:
Utilize the Main Controller and its methods to access all functions of the framework. Start by generating and adjusting the causal graph. Then, proceed to generate the dependencies.

4. **Sampling**: 
After completing the model generation, the model becomes ready for sampling operations. Proceed with sampling to obtain the sampled data. It is possible to successively modify dependencies or individual variable values within the graph, and continue sampling to incorporate these changes in the dataset. Additionally, there is an option to export the sampled data in CSV format.

## **Scripts**



1. **[config_objects.py](scripts/config_objects.py)**:
[Script for configuration setup](scripts/config_objects.py) 
- The script contains the configuration objects for the controllers. It gives the user the possibility to create new configrations by instantiating new objects. It is easy to use because all the relevant imports are already made, and the premade configurations are a good example


2. **[usage_ready_script.py](scripts/usage_ready_script.py)**:
- The script provides an extensive example on how to use the framework. The main controller is initialized and the main methods are provided to generate, manipulate and visualize,  first the causal graph and then the dependencies. The  In the last step an example for interactive sampling and export and acess to the data samples is shown.

3. **[usage_ready_script_simple.py](scripts/usage_ready_script_simple.py)**:
- an easy and fast example of graph and dependency generation and then sampling

4. **[comparisons_with_standard_and_real_graphs.py](scripts/comparisons_with_standard_and_real_graphs.py)**:
- This script enables the comparison of real graphs from the literature with random graphs generated by an Erdős-Rényi based graph model from the NetworkX library, as well as with graphs produced by our model. The comparison is conducted by sampling a thousand examples and using the average of various graph metrics provided by the NetworkX library.

## Demos

For a more detailed explanation on how to create configuration objects, and for an overview of the possible implementations and their parameters, please refer to the Documentation html.

[Documentations](./Documentation/_build/html/).



## Extensions

Extending the framework can involve a range of implementations, from adding minor features like new kernels, which bring more variety, to introducing entirely new approaches. A notable example of a substantial extension is in the EdgeMaker component. In this component, you can define the methods for edge generation within the graph.

The desired extensions include:

- Adding new kernels to enhance variety.
- Developing novel weight functions for calculating the probabilities of edge formation.
- Implementing a more advanced 'IEdgeMaker' that provides finer control over the distribution of edges.
- Creating a new 'IFunctionsMaker' that is capable of introducing discontinuities across multiple dimensions.
- Introducing innovative operators for the combination of different kernels.
