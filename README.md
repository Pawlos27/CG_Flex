# [CGFLEX]


## Table of Contents
1. [Description](#Description)
2. [Installation](#Installation)
3. [Feautures](#Feautures)
4. [Usage_Steps](#Usage_Steps)
4. [Usage_and_Scripts](#Usage_and_Scripts)
5. [Extensions](#Extensions)
6. [Remaining_Tasks](#Remaining_Tasks)


## Description
## Feautures

The Framwork consist of a Main controller and three sub controller , one for graph control the second for dependency control the third for sampling control.
The use of the framework is primarily through the main
controller and its methods. However, it is theoretically also possible to individually use sub controllers. 

The Frameworks involves 1. graph generation, 2. dependency generation, 3 sampling. 
It allows to check and adjust the model at each step of the generation process, while sampling it allows the manipulation of the dependencies in between sampling rounds.

The Framework involves 


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



## Usage_and_Steps

For the execution of the steps below premade scripts are provided.

The steps of the Usage of the framework:
1. **Preparation**:
 Initial setup. Create configuration objects for the sub-controllers and then the main controller( parameteriation and instantiation).
2. **Initiation**: 
Instatiate main controller with configuration object as parameter. 

3. **Model Generation**: 
Through the controller and its methods the user can access all functions of the framework First generate and adjust the causal graph. Subsequently,generate the dependencies.
4. **Sampling**: 
After model generation, the model is ready to perform sampling operations and 



## Usage_and_Scripts

2. **create_configs.py**:
- The script contains the configuration objects for the controllers. It gives the user the possibility to create new configrations by instantiating new objects. It is easy to use because all the relevant imports are already made, and the premade configurations are a good example


1. **usage_ready_script.py**:
- The script provides an extensive example on how to use the framework. The main controller is initialized and the main methods are provided to generate, manipulate and visualize,  first the causal graph and then the dependencies. The  In the last step an example for interactive sampling and export and acess to the data samples is shown.

2. **usage_ready_script_simple.py**:
- an easy and fast example of graph and dependency generation and then sampling

3. **experiments.py**:
    - Purpose: Facilitate the generation, training, and evaluation of neural network models.
    - Details: Provides utility functions to aid in machine learning experiments, including data loading, model
      creation, loss function assignment, and result saving.

## Extensions
## Remaining_Tasks

readme :
- optimization of explanation , images of graph, dependnecies, and nodelistobjects

parameters checks:
- input parameter controlls at config level

further implementations:
- visualization methods
- new parametrization layer for distributions
