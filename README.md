# [CGFLEX]


## Table of Contents
1. [Description](#Description)
2. [Installation](#Installation)
3. [Generate Class and Parameter Documentation](#Generate_Class_and_Parameter_Documentation)
4. [Feautures](#Feautures)
5. [Usage_Steps](#Usage_Steps)
6. [Usage_and_Scripts](#Usage_and_Scripts)
7. [Demos](#Demos)
8. [Extensions](#Extensions)
9. [Remaining_Tasks](#Remaining_Tasks)


## Description

This framework is designed to generate synthetic data for evaluating causal discovery algorithms. It addresses the challenge of unrealistic synthetic data by allowing for the controlled generation of datasets with known dependencies. This is crucial in fields where understanding causal relationships is vital, and real observational data often lack clear causal structures.

## Key Features

Graph Generation: Users can influence the graph's properties related to it´s shape, and interconnectivity, by setting various constraints like the number of nodes, edges, layers, indegree and also choosing the methodologies in which way the properties are .

Dependency Definition: The framework allows for the creation of dependencies by combining functions dependent on parent node values with an error term. It supports linear and nonlinear functions, as well as various error term types like normal, mixture, and conditional distributions.

Interactive Sampling: This feature enables users to adjust dependencies or fix variables between sampling rounds, facilitating the observation of effects on other variables. This enhances network investigation and is especially beneficial for algorithms designed to interact with the model.

- Advantages:

Provides a baseline for simulating realistic scenarios.
Enables gradual increase in complexity, challenging existing analytical methods.
Offers interactive model manipulation for deeper insights into variable interactions.
Ensures known dependencies for direct comparison with algorithmically identified dependencies, enhancing algorithm evaluation.
This framework is an essential tool for improving the quality of causal discovery algorithms, providing both flexibility and control needed for effective algorithm evaluation and performance comparison under diverse conditions.

## Framework Structure

The Framework consists of a Main Controller and three Sub Controllers, each specialized in a distinct area: the first for graph control, the second for dependency control, and the third for sampling control. This composite structure enables the Main Controller to oversee the Sub Controllers while acting as the central user interface.

The Main Controller is the primary interface through which users interact with the Framework, offering an accessible gateway to its functionalities. The Sub Controllers, functioning as coordinators, are pivotal in implementing the Framework's core functionalities. They manage and utilize all the respective subclasses, providing the basic capabilities through their methods.

In addition to this structure, the Framework is designed with high flexibility in mind. Through the use of interfaces, users can implement new classes to add or modify the Framework's functionalities, allowing for customization to meet specific needs and preferences.

Controller classes are initialized with configuration objects, which are instances of implemented classes parameterized during object creation. This design enables users to fine-tune the configurations of the data generation process, ensuring the Framework can adapt to a wide range of use cases.


## Installation

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


## **Generate_Class_and_Parameter_Documentation**

Inside the Documentation directory you can generate the documentation

simply run
```bash
make html
```

when using powershell you might need to use 

```bash
.\make.bat html
```

after making changes you can update the documentation files by running
```bash
sphinx-apidoc -o <Documentation_directory_path> <cg_flex_directory_path>
```

## **Usage_and_Steps**

For the execution of the steps below premade scripts are provided.

The steps of the Usage of the framework:

1. **Preparation**:
Begin with the initial setup. Create configuration objects for the Sub Controllers and then for the Main Controller. This involves parameterization and instantiation of these components.

2. **Initiation**:
Instantiate the Main Controller using the previously created configuration object as a parameter.

3. **Model Generation**:
Utilize the Main Controller and its methods to access all functions of the framework. Start by generating and adjusting the causal graph. Then, proceed to generate the dependencies.

4. **Sampling**: 
After completing the model generation, the model becomes ready for sampling operations. Proceed with sampling to obtain the sampled data. It is possible to successively modify dependencies or individual variable values within the graph, and continue sampling to incorporate these changes in the dataset. Additionally, there is an option to export the sampled data in CSV format.

## Usage_and_Scripts



1. **create_configs.py**:
- The script contains the configuration objects for the controllers. It gives the user the possibility to create new configrations by instantiating new objects. It is easy to use because all the relevant imports are already made, and the premade configurations are a good example


2. **usage_ready_script.py**:
- The script provides an extensive example on how to use the framework. The main controller is initialized and the main methods are provided to generate, manipulate and visualize,  first the causal graph and then the dependencies. The  In the last step an example for interactive sampling and export and acess to the data samples is shown.

3. **usage_ready_script_simple.py**:
- an easy and fast example of graph and dependency generation and then sampling

4. **comparisons_with_standard_and_real_graphs.py**:
- This script enables the comparison of real graphs from the literature with random graphs generated by an Erdős-Rényi based graph model from the NetworkX library, as well as with graphs produced by our model. The comparison is conducted by sampling a thousand examples and using the average of various graph metrics provided by the NetworkX library.

## Demos

For a more detailed explanation on how to create configuration objects, and for an overview of the possible implementations and their parameters, please refer to [Demo_configuration.md](./Demo_configuration.md).

For a more detailed explanation on how to use the framework, see [Demo_usage.md](./Demo_usage.md).


## Extensions

Extending the framework can involve a range of implementations, from adding minor features like new kernels, which bring more variety, to introducing entirely new approaches. A notable example of a substantial extension is in the EdgeMaker component. In this component, you can define the methods for edge generation within the graph.

The desired extensions include:

- Adding new kernels to enhance variety.
- Developing novel weight functions for calculating the probabilities of edge formation.
- Implementing a more advanced 'IEdgeMaker' that provides finer control over the distribution of edges.
- Creating a new 'IFunctionsMaker' that is capable of introducing discontinuities across multiple dimensions.
- Introducing innovative operators for the combination of different kernels.

## Remaining_Tasks

readme :
- optimization of explanation , demo.md file with images of graph, dependnecies, and nodelistobjects

parameters checks:
- input parameter controlls at config level

further implementations:
- visualization methods
- new parametrization layer for distributions
