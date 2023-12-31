


Welcome to CGFlex's documentation!
==================================

CGFlex is a framework is designed to generate synthetic data for evaluating causal discovery algorithms.
In this documentation you can find information about the classes and its parameters.
The Framework has the following composite structure:
The Main Controller controls 3 sub controller classes: 1.Graphmaker 2.Dependencymaker 3.Sampler.
The Sub controller classes coordinate methods of further classes.
Instantiations of these classes are provided to the controllers via configuration objects. 
In this objects the user can set various parameter of the classes. 
This documentation provides an overview over al classes and also information about the parameters and its effects.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   cgflex_project.Main_Controller
   cgflex_project.Module_Dependencymaker
   cgflex_project.Module_Graphmaker
   cgflex_project.Module_Sampler
   cgflex_project.Shared_Classes


Further ressources:
-------------------

- You can find more Information about the usage of the framework in the readme files

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
