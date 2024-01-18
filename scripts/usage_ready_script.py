from config_objects import  config_main_controller_default
from cgflex_project.Main_Controller.Controllermodule import Cg_flex_controller






#######################
### 1.Set Config!!  ### 
#######################
config = config_main_controller_default # default config, for own config use config_objects.py script


#######################
###  2.Set Stages!! ### --> Remove the # in front of the STage you want to use to set the Stage , the run code.
#######################

Stage = 3.5

##Gaphgeneration##
#Stage = 1      # Initiates Controller and Generates Graph
#Stage = 1.5     # Visualize/Check  Graph
#Stage = 1.55   # Changes Graph and Safe

##Dependencygeneration and Sampling##
#Stage = 2      # Loads Generates Dependendencies
#Stage = 2.5    # Visualize Dependencies
#Stage = 2.51    # Visualize Dependencies between certain nodes
#Stage = 2.55   # Change Dependendencies and Safe

#Stage = 3     # Sampling and Visualisation
#Stage = 3.1   # Sampling after setting sources to tsd/fixed_values/distributions
#Stage = 3.1   # Sampling after setting sources to tsd/fixed_values/distributions
#Stage = 3.5   # Making Changes to Dependencies and sample again






if Stage == 1:
    dag_controll = Cg_flex_controller(config=config)
    dag_controll.make_graphs(number_of_decoupled_elements=1)
    dag_controll.plot_graph()
    dag_controll.plot_graph_by_layer()
    dag_controll.print_nodelists_graph()
    dag_controll.print_graph_metrics()
    dag_controll.safe_controller_state(file_path=None,file_name="full_state")

if Stage == 1.5:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.plot_graph()
    dag_controll.print_graph_metrics()


if Stage == 1.55:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.plot_graph()
    #dag_controll.set_new_sinks_graph()
    #dag_controll.plot_graph(graph_ids=[0])
    #dag_controll.set_new_sources_graph(list_of_sources=[2,3])
    dag_controll.set_new_edges_graph(graph_id=0)
    dag_controll.plot_graph()
    dag_controll.print_graph_metrics()
    dag_controll.safe_controller_state(file_name="full_state")


if Stage == 2:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.make_dependencies()
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3],visualized_dimensions=(0,1))
    dag_controll._get_total_output_range()
    dag_controll.safe_controller_state(file_path=None,file_name="full_state")


if Stage == 2.5:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    #dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4,5,6],visualized_dimensions=(0,1))
    #dag_controll.show_dependency_from_one_node(node_id_target=4, node_id_dependency=2,graph_id=0)
    #dag_controll.show_dependency_from_2_nodes(node_id_dependency_x=1,node_id_dependency_y=2,node_id_target=4, graph_id=0)
    #dag_controll.show_dependency_from_parents_scatterplot(node_id_target=6, resolution=20)
    #dag_controll.show_dependency_functions_only(ids=[0,1,2],)
    dag_controll.show_dependency_errorterm_only(ids=[0,1,2,3,4,5,6,7])

if Stage == 2.55:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    #dag_controll.reset_dependencies()
    #dag_controll.reset_dependencies_specific(node_ids=[0,1],graph_id=0)
    #dag_controll.replace_all_dependencies_by_initial_distributions()
    #dag_controll.replace_dependencies_by_initial_distributions(node_ids=[0,1], graph_id=0)
    #dag_controll.replace_dependencies_by_initial_distributions_abstract_id(ids_abstract=[0,1])
    #dag_controll.replace_dependencies_by_single_values(node_ids_and_values=[(0,0),(1,0)])
    #dag_controll.replace_dependencies_by_single_values_abstract_id(abstract_ids_and_values=[(0,1),(1,1)])
    #dag_controll.replace_dependencies_by_single_values_from_random_distribution(node_ids=[0,1,2], graph_id=0)
    #dag_controll.replace_dependencies_by_single_random_values_abstract_id(ids_abstract=[0,1,2])
    #dag_controll.replace_dependencies_by_tsd_function(node_ids=[0,1,2],graph_id=0)
    dag_controll.replace_dependencies_by_tsd_function_abstract_id(ids_abstract=[0,1])
    dag_controll.safe_controller_state(file_name="full_state")
                                            

if Stage == 3:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.sample_values_full(number_of_samples=50)
    dag_controll.show_values_histogramm_raw(node_id=[0,1,2,3], graph_id=0)
    dag_controll.show_values_histogramm_abstracted(id_abstract=[0,1,2,3])
    dag_controll.export_value_id_samples_abstract()
    dag_controll.print_samples_raw()
    a, b =dag_controll.find_graph_id_and_node_id_from_shuffle_index(id_abstract=0)
    print(a)
    dag_controll.safe_controller_state(file_name="full_state")


if Stage == 3.1:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.reset_samples()
    dag_controll.replace_id_shuffle_index()
    dag_controll.reset_tsd_counter()
    dag_controll.set_sources_as_tsd_function()
    #dag_controll.set_sources_as_distributions()
    #dag_controll.set_sources_to_fixed_values()
    dag_controll.sample_values_full(number_of_samples=50)



if Stage == 3.5:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.sample_with_replaced_values(replaced_nodes=[[(0,1),(2,1)]],number_of_samples=5)
    dag_controll.show_values_histogramm_raw(node_id=0)





