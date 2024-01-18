from config_objects import  config_main_controller_default
from cgflex_project.Main_Controller.Controllermodule import Cg_flex_controller, Graph_controller


config = config_main_controller_default
#Stage = 1
#Stage = 1.5
#Stage = 2
#Stage = 2.0012
#Stage = 2.11
#Stage = 2.5
#Stage = 2.0101
Stage = 10




if Stage == 1:
    dag_controll = Cg_flex_controller(config=config)
    dag_controll.make_graphs(number_of_decoupled_elements=1)
    dag_controll.plot_graph()
    dag_controll.print_graph_metrics()
    dag_controll.safe_controller_state(file_path=None,file_name="full_state")


if Stage == 1.5:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    #dag_controll.set_new_sinks_graph()
    dag_controll.plot_graph()
    dag_controll.set_new_sources_graph(list_of_sources=[41])
    dag_controll.plot_graph()


if Stage == 2:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.make_dependencies()
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,10,11,12,13,14,16,17,18])
    dag_controll.safe_controller_state(file_path=None,file_name="full_state")
    dag_controll._get_total_output_range()
    dag_controll.sample_with_replaced_values(replaced_nodes=[[(0,1),(2,1)]],number_of_samples=5)
    dag_controll.show_values_histogramm_raw(node_id=0)

if Stage == 10:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    #dag_controll.sample_with_replaced_values(replaced_nodes=[[(0,1),(2,1)]],number_of_samples=5)
    dag_controll.sample_with_replaced_values_abstract_ids(replaced_nodes_abstract_id=[(0,1),(2,1)],number_of_samples=5)
    dag_controll.show_values_histogramm_abstracted(id_abstract=[0,2])
    #dag_controll.show_values_histogramm_raw(node_id=1)
if Stage == 2.0012:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.print_nodelists_graph()
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[12], visualized_dimensions=(0,1))
    #dag_controll.sample_values_full(number_of_samples=2)
    #dag_controll.print_samples_raw()
    dag_controll.show_dependency_from_parents_scatterplot(node_id_target=12,resolution=20, visualized_dimensions=(0,1))
    #samples = dag_controll.return_samples_abstracted_hidden_nodes(num_nodes_to_hide=30)
    #print(samples)


if Stage == 2.0101 :
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.plot_graph()
    dag_controll.show_dependency_from_one_node(node_id_target= 18, node_id_dependency=13)
    dag_controll.show_dependency_from_2_nodes(node_id_target= 25,node_id_dependency_y=21,node_id_dependency_x=20)

    dag_controll.show_dependencies(ids=[18])
    dag_controll.show_dependencies(ids=[25])


if Stage == 2.11:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4,5,6,7,10,11])

if Stage == 2.22:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.set_sources_as_distributions()
    dag_controll.sample_values_full(number_of_samples=100)
    print(dag_controll.samples_raw)
    print(dag_controll.samples_abstracted_id)
    print(dag_controll.samples_accumulated_per_id_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=[0,1,2,3,4,5,6,7,8,9,10,11])






if Stage == 2.001:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.show_dependencies(ids=[15,18,25,50,61])


if Stage == 2.01:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    sources=dag_controll.get_sourcelist_dependencymaker()
    print(sources)
    dag_controll.set_sources_as_tsd_function()
    dag_controll.show_dependencies(ids=[0,1,6,7, 15])
    dag_controll.sample_values_full(number_of_samples=20)
    dag_controll.show_values_histogramm_raw(node_id=0)
    dag_controll.show_values_histogramm_raw(node_id=6)
    dag_controll.show_values_histogramm_raw(node_id=7)
    print(dag_controll.samples_raw)



if Stage == 2.1:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4,5,6,7,8,9])
  
if Stage == 2.2:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.sample_values_full(number_of_samples=3)
    dag_controll.export_value_id_samples_abstract()
    print(dag_controll.samples_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=15)
    dag_controll.show_values_histogramm_abstracted(id_abstract=4)
    dag_controll.show_values_histogramm_abstracted(id_abstract=8)








if Stage == 2.5:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.sample_values_full(number_of_samples=30)
    dag_controll.export_value_id_samples_abstract()
    dag_controll.show_values_histogramm_abstracted(id_abstract=8)



if Stage == 2.6:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    sources = dag_controll.get_sourcelist_dependencymaker()
    print(f"sources are at {sources}")
    dag_controll.set_sources_as_dependencies()
    dag_controll.sample_values_full(number_of_samples=20)
    print(dag_controll.samples_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=12)
    dag_controll.show_values_histogramm_abstracted(id_abstract=0)
    dag_controll.show_values_histogramm_abstracted(id_abstract=1)

if Stage == 2.7:
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state")
    dag_controll.show_values_histogramm_abstracted(id_abstract=10)
if Stage == 3.0:
    pass
