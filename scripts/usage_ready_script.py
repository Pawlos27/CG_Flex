from config_objects import blueprint_controller_test
from cgflex_project.Main_Controller.Controllermodule import Cg_flex_controller




Stage = 1




if Stage == 1:
    dag_controll = Cg_flex_controller(config=blueprint_controller_test)
    dag_controll.make_graphs(number_of_decoupled_elements=1)
    dag_controll.plot_graph()
    dag_controll.safe_controller_state(file_path=None,file_name="full_state")


if Stage == 1.5:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    #dag_controll.set_new_sinks_graph()
    dag_controll.plot_graph()


if Stage == 2:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.make_dependencies()
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4,5,6,7,10,11,12,13,14,15,16])
    dag_controll.safe_controller_state(file_path=None,file_name="full_state")
    dag_controll._get_total_output_range()

if Stage == 2.11:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4,5,6,7,10,11,12,13,14,15,16])

if Stage == 2.22:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.set_sources_as_distributions()
    dag_controll.sample_values_full(number_of_samples=100)
    print(dag_controll.samples_raw)
    print(dag_controll.samples_abstracted_id)
    print(dag_controll.samples_accumulated_per_id_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=[0,1,2,3,4,5,6,7,8,9,10,11])






if Stage == 2.001:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.show_dependencies(ids=[15,18,25,50,61])


if Stage == 2.01:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
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
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4,5,6,7,8,9])
  
if Stage == 2.2:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.sample_values_full(number_of_samples=3)
    dag_controll.export_value_id_samples_abstract()
    print(dag_controll.samples_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=15)
    dag_controll.show_values_histogramm_abstracted(id_abstract=4)
    dag_controll.show_values_histogramm_abstracted(id_abstract=8)
    print(dag_controll.samples_accumulated_per_id_raw)







if Stage == 2.5:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.sample_values_full(number_of_samples=30)
    dag_controll.export_value_id_samples_abstract()
    print(dag_controll.samples_raw)
    print(dag_controll.samples_abstracted_id)
    print(dag_controll.samples_accumulated_per_id_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=8)

if Stage == 2.55:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.set_sources_as_distributions()
    dag_controll.sample_values_full(number_of_samples=100)
    dag_controll.export_value_id_samples_abstract()
    print(dag_controll.samples_raw)
    print(dag_controll.samples_abstracted_id)
    print(dag_controll.samples_accumulated_per_id_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=0)
    dag_controll.show_values_histogramm_abstracted(id_abstract=1)
    dag_controll.show_values_histogramm_abstracted(id_abstract=2)
    dag_controll.show_values_histogramm_abstracted(id_abstract=3)
    dag_controll.show_values_histogramm_abstracted(id_abstract=8)
    dag_controll.show_values_histogramm_abstracted(id_abstract=9)
    dag_controll.show_values_histogramm_abstracted(id_abstract=12)
    dag_controll.show_values_histogramm_abstracted(id_abstract=13)
    dag_controll.show_values_histogramm_abstracted(id_abstract=14)
    dag_controll.show_values_histogramm_abstracted(id_abstract=15)


if Stage == 2.6:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    sources = dag_controll.get_sourcelist_dependencymaker()
    print(f"sources are at {sources}")
    dag_controll.set_sources_as_dependencies()
    dag_controll.sample_values_full(number_of_samples=20)
    print(dag_controll.samples_raw)
    dag_controll.show_values_histogramm_abstracted(id_abstract=12)
    dag_controll.show_values_histogramm_abstracted(id_abstract=0)
    dag_controll.show_values_histogramm_abstracted(id_abstract=1)

if Stage == 2.7:
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state")
    dag_controll.show_values_histogramm_abstracted(id_abstract=10)
if Stage == 3.0:
    pass
