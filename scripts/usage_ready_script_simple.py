from config_objects import config_main_controller_no_errorterm, config_main_controller_default
from cgflex_project.Main_Controller.Controllermodule import Cg_flex_controller




config = config_main_controller_default
Stage = 1 #Initiates Controller generates graph, dependencies and samples
#Stage = 1.5 # Visual dependencies
#Stage = 2
#Stage = 2.5
#Stage = 3

if Stage == 1:
    #generate graph
    dag_controll = Cg_flex_controller(config=config)
    dag_controll.make_graphs(number_of_decoupled_elements=1)
    dag_controll.plot_graph()
    # generate dependency
    dag_controll.make_dependencies()
    #dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0])
    dag_controll.show_dependencies(ids=[0,1,2,3])

    #sampling
    dag_controll.sample_values_full(number_of_samples=100)
    #visualize samples for certain id´s
    dag_controll.show_values_histogramm_abstracted(id_abstract=[0,1,2])
    #safe controller state with model and samples
    dag_controll.safe_controller_state(file_path=None, file_name="full_state_simple")

    samples_abstracted = dag_controll.samples_abstracted_id
    samples_raw = dag_controll.samples_raw
    #print(samples_abstracted)

if Stage == 1.5 :
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state_simple")
    dag_controll.show_dependencies_enforced_3d_visualisation(ids=[0,1,2,3,4])


if Stage == 2 :
    dag_controll = Cg_flex_controller.load_controller_state(config=config,file_path=None, file_name="full_state_simple")
    

    dag_controll.set_sources_as_tsd_function()
    dag_controll.show_dependencies(ids=[0])
    dag_controll.sample_values_full(number_of_samples=20)
    dag_controll.show_values_histogramm_raw(node_id=0)
    dag_controll
    samples_abstracted = dag_controll.samples_abstracted_id
    samples_raw = dag_controll.samples_raw

    #print(samples_raw)



