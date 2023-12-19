from config_objects import blueprint_controller_test
from cgflex_project.Main_Controller.Controllermodule import Cg_flex_controller


Stage = 1.5

if Stage == 1:
    dag_controll = Cg_flex_controller(config=blueprint_controller_test)
    dag_controll.make_graphs(number_of_decoupled_elements=1)
    dag_controll.plot_graph()

    dag_controll.make_dependencies()
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0])
    dag_controll.show_dependencies(ids=[0,1,2,3])

    dag_controll.sample_values_full(number_of_samples=100)
    dag_controll.show_values_histogramm_abstracted(id_abstract=[0,1,2])
    dag_controll.safe_controller_state(file_path=None, file_name="full_state_simple")

    samples_abstracted = dag_controll.samples_abstracted_id
    samples_raw = dag_controll.samples_raw

    print(samples_abstracted)

if Stage == 1.5 :
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state_simple")
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4])
    dag_controll.show

if Stage == 2 :
    dag_controll = Cg_flex_controller.load_controller_state(config=blueprint_controller_test,file_path=None, file_name="full_state_simple")

    sources=dag_controll.get_sourcelist_dependencymaker()
    print(sources)

    dag_controll.set_sources_as_tsd_function()
    dag_controll.show_dependencies(ids=[0])
    dag_controll.sample_values_full(number_of_samples=20)
    dag_controll.show_values_histogramm_raw(node_id=0)
    print(dag_controll.samples_raw)



