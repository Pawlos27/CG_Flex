from config_objects import blueprint_controller_test, blueprint_test
from cgflex_project.Main_Controller.Controllermodule import Cg_flex_controller, Graph_controller
import random





import os
import networkx as nx
from pgmpy.readwrite import BIFReader
import matplotlib.pyplot as plt  # Import Matplotlib


def generate_random_dag(num_nodes, edge_probability, counter, Modus):
    if Modus == "random":
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for node in G.nodes():
            for potential_child in range(node+1, num_nodes):
                if random.random() < edge_probability:
                    G.add_edge(node, potential_child)
    elif Modus == "fix":
        if counter == 1:
            file_name= "alarm.bif"
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)
            reader = BIFReader(file_path)
        if counter == 2:
            file_name= "barley.bif"
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)
            reader = BIFReader(file_path)
        if counter == 3:
            file_name= "child.bif"
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)
            reader = BIFReader(file_path)
        if counter == 4:
            file_name= "insurance.bif"
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)
            reader = BIFReader(file_path)
        if counter == 5:
            file_name= "alarm.bif"
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_folder, file_name)
            reader = BIFReader(file_path)
        model = reader.get_model()
        nodes = model.nodes()
        edges = model.edges()
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
    
    else :
        Graph_generator = Graph_controller(config=blueprint_test)
        Graph_generator.make_graph()
        edges = Graph_generator.get_edgelist()
        G = nx.DiGraph()
        G.add_edges_from(edges)

    return G

def calculate_metrics(G):
    # Berechnen der Quellen und Senken
    sources = [node for node, deg in G.in_degree() if deg == 0]
    sinks = [node for node, deg in G.out_degree() if deg == 0]

    metrics = {
        "number_of_nodes": G.number_of_nodes(),
        "number_of_edges": G.number_of_edges(),
        #"average_shortest_path_length": nx.average_shortest_path_length(G),
        "density": nx.density(G),
        "longest_path_length": len(nx.dag_longest_path(G)),
        "max_degree": max(G.degree(), key=lambda item: item[1])[1],
        "average_in_degree": sum(dict(G.in_degree()).values()) / G.number_of_nodes(),
        "average_out_degree": sum(dict(G.out_degree()).values()) / G.number_of_nodes(),
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "max_closeness": max(nx.closeness_centrality(G).items(), key=lambda x: x[1])[1],
        "max_betweenness": max(nx.betweenness_centrality(G).items(), key=lambda x: x[1])[1],
        "number_of_sources": len(sources),
        "number_of_sinks": len(sinks)
    }
    return metrics

def average_metrics(metrics_list):
    avg_metrics = {key: sum(metric[key] for metric in metrics_list) / len(metrics_list) for key in metrics_list[0]}
    return avg_metrics


# Parameters
Modus = "cg_flex"


num_graphs = 5  # Number of random DAGs to generate
if Modus == "fix":
   num_graphs = 5  
num_nodes = 10   # Number of nodes in each DAG
edge_probability = 0.2  # Probability of edge creation
counter = 1

# Generate graphs and calculate metrics
all_metrics = []
for _ in range(num_graphs):
    
    G = generate_random_dag(num_nodes, edge_probability, counter=counter, Modus=Modus)
    metrics = calculate_metrics(G)
    all_metrics.append(metrics)
    counter =+1

# Average the metrics
average_of_metrics = average_metrics(all_metrics)

# Print the average metrics
for metric, value in average_of_metrics.items():
    print(f"Average {metric}: {value}")

Test = "a"



graph= Graph_controller(config=blueprint_test)
#graph.make_graph()
#graph.showgraph()


