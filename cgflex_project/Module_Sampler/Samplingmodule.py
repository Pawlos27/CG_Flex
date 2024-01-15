
import cgflex_project.Module_Dependencymaker._dependencymaker as _dependencymaker
from cgflex_project.Shared_Classes.NodeObject import Nodeobject
import cgflex_project.Shared_Classes.distributions as distributions
from typing import Any, List, Type, Tuple, Optional
import cgflex_project.Module_Dependencymaker.Dependencymodule as Dependencymodule
import random
from cgflex_project.Shared_Classes.blueprints import Blueprint_graph, Blueprint_dependency, Blueprint_sampling
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from  cgflex_project.Module_Dependencymaker._dependencymaker_tsd_functions import ITsd_functions




class Sampling_controller:
    def __init__(self,config:Blueprint_sampling):
        self.nodelist_nested_list = None
        self.id_value_list_of_arrays  = None # list[sample_id][graph_id][node_id][=id,=value]] multi nested list of arrays, first level of list is representing the sample number, second level is representing the graph_id, then on the third level  for each graph there is an 2s array of value_id_pairs
        self.samples_accumulated_per_id = None
        self.id_shuffle_index = None
        self.samples_abstracted_id = None
        self.config = config
        self.tsd_counter = 0

    def load_nodelist_dependencies(self, nodelist_nested: List[List[Type[Nodeobject]]]): 
        self.nodelist_nested_list = nodelist_nested

    def reset_config(self,config:Blueprint_sampling):
        self.config = config

    def reset_tsd_counter(self):
        self.tsd_counter = 0

    def reset_samples(self):
        self.id_value_list_of_arrays = None
        self.samples_abstracted_id = None
        self.samples_accumulated_per_id = None
    
    def print_samples_raw(self):
        if self.id_value_list_of_arrays == None:
            raise ValueError("there are no samples yet to print, please generate samples first")
        data=[]
        counter_sample = 0
        for sample in self.id_value_list_of_arrays:
                counter_graph = 0
                for graph in sample:
                    for node in graph :

                        datapoint = {"Sample_number": counter_sample,'Graph_id': counter_graph, "Node_id": int(node[0]) ,'Value': node[1]}
                        data.append(datapoint)
                    counter_graph +=1
                counter_sample += 1     
        df=pd.DataFrame(data)
        print (df)



    def return_nested_nodelist(self):
        return self.nodelist_nested_list

    def return_samples_raw(self):
        return self.id_value_list_of_arrays
    
    def return_samples_accumulated_per_id(self):
        return self.samples_accumulated_per_id 
    
    def return_samples_abstracted_id (self):
        return self.samples_abstracted_id 
    
    def return_samples_abstracted_hidden_nodes(self, num_nodes_to_hide:int=5):
        if self.samples_abstracted_id is None:
            raise ValueError("No abstracted samples existing yet")
        if num_nodes_to_hide >= len(self.samples_abstracted_id[0]):
            raise ValueError("you try to hide more nodes then existing")
        shortened_samples = []
        for sample in self.samples_abstracted_id:
            shortened_sample = sample[:len(sample) - num_nodes_to_hide]
            shortened_samples.append(shortened_sample)
        return np.array(shortened_samples)

    def get_values_from_nodelist(self,graph_id:int, node_ids:List[int]):
        value_list=[]
        for node_id in node_ids:
            value = self.nodelist_nested_list[graph_id][node_id].value
            value_list.append(value)
        return value_list

    def get_value_from_node(self,graph_id:int, node_id: int ):
        value = self.nodelist_nested_list[graph_id][node_id].value
        return value

    def _calculate_values_per_graph(self, graph_id): # calculate values only for targeted nodelist
            counter = 0
            for node in self.nodelist_nested_list[graph_id]:
                if isinstance(node.dependency, (float, int)):
                    node.value = node.dependency
                elif isinstance(node.dependency, distributions.IDistributions):
                    node.value = node.dependency.get_value_from_distribution()
                elif isinstance(node.dependency, _dependencymaker.Dependencies):
                    #if node.source == True:
                        #node.value= node.dependency.calculate_normalized_value(x_values=[time_tick])
                    #else :
                    node.value= node.dependency.calculate_normalized_value(x_values=self.get_values_from_nodelist(graph_id=graph_id,node_ids=node.parents))
                elif isinstance(node.dependency, ITsd_functions):
                    node.value = node.dependency.calculate_value(x=self.tsd_counter)
                
                counter +=1

    def calculate_values_all_graphs(self):
        graph_number = len(self.nodelist_nested_list)
        for i in range(graph_number):
            self._calculate_values_per_graph(graph_id=i)

    def _calculate_value_up_to_certain_node_and_replace_values(self,node_id:int, replaced_nodes:List[Tuple[int,float]] , graph_id:int=0 , replaced_nodes_to_zero:List[int]=[]):
        target_value = None

        for node in self.nodelist_nested_list[graph_id]:
            if isinstance(node.dependency, (float, int)):
                node.value = node.dependency
            elif isinstance(node.dependency, distributions.IDistributions):
                node.value = node.dependency.get_value_from_distribution()
            elif isinstance(node.dependency, _dependencymaker.Dependencies):
                node.value= node.dependency.calculate_normalized_value(x_values=self.get_values_from_nodelist(graph_id=graph_id,node_ids=node.parents))
            elif isinstance(node.dependency, ITsd_functions):
                node.value = node.dependency.calculate_value(x=self.tsd_counter)
            # replace values for specific nodes
            for replaced_node in replaced_nodes:
                if replaced_node[0] == node.id:
                    node.value = replaced_node[1]
            for replaced_node in replaced_nodes_to_zero:
                if replaced_node == node.id:
                    node.value = 0
            if node.id == node_id:
                target_value = node.value
                break
        return target_value
           
    def show_dependency_from_one_node(self, node_id_target:int, node_id_dependency:int,range: Tuple[float,float], graph_id=0 , resolution:int=100):
        # Check if node_id_target is greater than node_id_dependency_
        if node_id_target <= node_id_dependency:
            raise ValueError("node_id_target must be greater than node_id_dependency_")
    
        # calculate values
        input_values = np.linspace(range[0], range[1], resolution)
        output_values = []
        for input in input_values:
            # set node to input value
            value = self._calculate_value_up_to_certain_node_and_replace_values(node_id=node_id_target, graph_id=graph_id, replaced_nodes=[(node_id_dependency,input)] )
            output_values.append(value)

        plt.scatter(input_values, output_values)
        plt.xlabel(f'Input Values - Node: {node_id_target}')
        plt.ylabel(f'Output Values - Node: {node_id_dependency}')
        plt.title(f'Scatter Plot of dependency  of Node: {node_id_target} from Node: {node_id_dependency}')
        plt.grid(True)
        plt.show()
            
    def show_dependency_from_2_nodes(self, node_id_target:int, node_id_dependency_x:int,node_id_dependency_y:int, range_f:Tuple[float,float], graph_id:int=0 , resolution:int=10, replaced_nodes_to_zero:List[int]=[]):
        if node_id_target <= node_id_dependency_x:
            raise ValueError("node_id_target must be greater than node_id_dependency_x")
        if node_id_target <= node_id_dependency_y:
            raise ValueError("node_id_target must be greater than node_id_dependency_y")

        input_range = np.linspace(range_f[0], range_f[1], resolution)
        # Generate meshgrid for input values
        X, Y = np.meshgrid(input_range, input_range)

        # Initialize an empty list to store output values
        Z = []

        # Calculate output for each pair in the meshgrid
        for i in range(len(input_range)):
            for j in range(len(input_range)):
                x_val = X[i, j]
                y_val = Y[i, j]
                z_val = self._calculate_value_up_to_certain_node_and_replace_values(node_id=node_id_target, graph_id=graph_id, replaced_nodes=[(node_id_dependency_x,x_val),(node_id_dependency_y,y_val)], replaced_nodes_to_zero=replaced_nodes_to_zero )
                Z.append(z_val)

        # Convert Z to a numpy array
        Z = np.array(Z).reshape(X.shape)
        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z)

        ax.set_xlabel(f'Node_x: {node_id_dependency_x}' )
        ax.set_ylabel(f'Node_y: {node_id_dependency_y}')
        ax.set_zlabel(f'Output: Node {node_id_target} ')

        plt.title(f'Scatter Plot of dependency  of Node: {node_id_target} from Node_x: {node_id_dependency_x} and from Node_y: {node_id_dependency_y}')
        plt.show()

    def show_dependency_from_parents_scatterplot(self, node_id_target:int, range_f:Tuple[float,float], graph_id:int=0, resolution:int=100, visualized_dimensions:Tuple[int,int] = (0,1)):
        if min(visualized_dimensions) < 0:
            raise ValueError("dimension cannot be smaller then 0")
        parents = self._get_parents_of_node(node_id=node_id_target, graph_id=graph_id)
        parents_copy = list(parents)
        if len(parents) == 1:
            self.show_dependency_from_one_node(node_id_target=node_id_target, range=range_f, node_id_dependency=parents[0], graph_id=graph_id, resolution=resolution)
        elif len(parents_copy)> 1:
            if max(visualized_dimensions) > len(parents_copy):
                node_id_dependency_x = parents_copy.pop(0)
                node_id_dependency_y = parents_copy.pop(1)
            else:
                node_id_dependency_x = parents_copy.pop(visualized_dimensions[0])
                node_id_dependency_y = parents_copy.pop(visualized_dimensions[1])          
            self.show_dependency_from_2_nodes(node_id_target=node_id_target, node_id_dependency_x=node_id_dependency_x, node_id_dependency_y=node_id_dependency_y,range_f=range_f, graph_id=graph_id,resolution=resolution, replaced_nodes_to_zero=parents_copy)
        else:
            print(f"node_id:{node_id_target} has no parents, hence there is no dependency for scatterplot visualisation")

    def _get_parents_of_node(self,node_id:int, graph_id:int=0):
        parents = self.nodelist_nested_list[graph_id][node_id].parents
        return parents

    def _make_value_id_samples_array(self): # makes list of id_value_pair arrays, each sub graph has an 2d array
        id_values_list_nested_one_sample = [] 
        for nodelist in self.nodelist_nested_list:
            id_value_list_graph = []
            for node in nodelist:
                id_value_pair = [int(node.id),node.value]
                id_value_list_graph .append(id_value_pair)
            id_values_list_nested_one_sample.append(np.array(id_value_list_graph ))
        return id_values_list_nested_one_sample

    def sample_value_id_pairs(self, number_of_samples=1):
        if self.id_value_list_of_arrays == None:
            self.id_value_list_of_arrays = []
        

        for i in range(number_of_samples):
            self.calculate_values_all_graphs()
            list_of_samples = self._make_value_id_samples_array()
            self.id_value_list_of_arrays.append(list_of_samples)
            #counter for tsd functions, resetable
            self.tsd_counter += 1
    
    def reset_samples(self):
        self.id_value_list_of_arrays  = None 
        self.samples_accumulated_per_id = None
        self.samples_abstracted_id = None 

    def make_accumulated_samples(self): # sorting all sampled values to the linked node_id, search algorhytm is matching structure of nested_id_pairs
        list_of_nested_accumulated_arrays = []
        number_of_samples = len(self.id_value_list_of_arrays)
        number_of_graphs = len(self.id_value_list_of_arrays[0])
        for g in range(number_of_graphs):
            list_of_nested_accumulated_arrays_per_graph = []
            component_lenght = len(self.id_value_list_of_arrays[0][g])
            for i in range(component_lenght):
                samples_container_node = []
                id_node = i
                for j in range(number_of_samples):
                    sample_value = self.id_value_list_of_arrays[j][g][i][1]
                    samples_container_node.append(sample_value)
                id_and_value = [id_node,samples_container_node]
                list_of_nested_accumulated_arrays_per_graph.append(id_and_value)
            list_of_nested_accumulated_arrays.append(list_of_nested_accumulated_arrays_per_graph)
        self.samples_accumulated_per_id = list_of_nested_accumulated_arrays           

    def print_values(self):
        data=[]
        for nodelist in self.nodelist_nested_list:
            for node in nodelist:
                node_value = {'ID': node.id, 'value': node.value}
                data.append(node_value)
        df=pd.DataFrame(data)
        print (df)
        
    

    def show_values_histogramm(self, graph_id:int ,node_id:int, output_range):

        data = self.samples_accumulated_per_id[graph_id][node_id][1]

        #  the actual range needed based on data and output_range
        data_min, data_max = min(data), max(data)
        range_min = min(data_min, output_range[0])
        range_max = max(data_max, output_range[1])

        # histogram
        plt.hist(data, bins=100, range=(range_min, range_max), edgecolor='black')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(f"Histogram of values for node {node_id} in graph {graph_id}")
        plt.show()

    def make_new_id_shuffle_index(self):
        id_index_dictionary_list = []
        number_components = len(self.id_value_list_of_arrays[0])
        counter = 0
        for j in range(number_components):
            for id_value_pair in self.id_value_list_of_arrays[0][j]:
                dict_id_map = {"graph_id": int(j), "node_id": int(id_value_pair[0]), "abstract_id": int(counter)}
                id_index_dictionary_list.append(dict_id_map)
                counter +=1
        
        shuffle_ids = list(range(counter)) # we create the id´s for the shuffle and then fill them into the dictionary
        random.shuffle(shuffle_ids)
        for i in range (len(shuffle_ids)):
            id_index_dictionary_list[i]["abstract_id"] = shuffle_ids[i]
        self.id_shuffle_index = id_index_dictionary_list

    def replace_id_shuffle_index(self):

        shuffle_value_ids = list(range(len(self.id_shuffle_index))) # we create the id´s for the shuffle and then fill them into the dictionary
        random.shuffle(shuffle_value_ids)
        for i in range (len(shuffle_value_ids)):
            self.id_shuffle_index[i]["value_id"] = shuffle_value_ids[i]

    def find_graph_id_and_node_id_from_shuffle_index(self,id_abstract:int) -> Tuple[int, int]:
        shuffle_index = self.id_shuffle_index
        if shuffle_index == None:
            raise ValueError("there is no shuffle_id index yet, or the nodelists are not loaded into the sample controlelr yet")
        else:
            shuffle_index_sorted= sorted(shuffle_index, key=lambda x: x["abstract_id"])
            graph_id = shuffle_index_sorted[id_abstract]["graph_id"]
            node_id = shuffle_index_sorted[id_abstract]["node_id"]
            return graph_id, node_id

    def make_abstracted_samples(self):
        if self.id_shuffle_index == None:
            self.make_new_id_shuffle_index()
        id_value_list_abstracted = []

        number_of_samples = len(self.id_value_list_of_arrays)
        for j in range(number_of_samples):
            one_sample = []
            for dict in self.id_shuffle_index:
                graph_id = dict["graph_id"]
                node_id =dict["node_id"]
                node_id = int(node_id)
                id_value_pair_abstracted = [int(dict["abstract_id"]),self.id_value_list_of_arrays[j][graph_id][node_id][1]] # base ids are identical with indices
                one_sample.append(id_value_pair_abstracted)
            one_sample_sorted = sorted(one_sample, key=lambda x: x[0])
            id_value_list_abstracted.append(one_sample_sorted)
        self.samples_abstracted_id = np.array(id_value_list_abstracted)
                 
    def export_value_id_samples_abstract(self,  filename="last_export", filepath=None):
        data_list=self._make_data_for_export()
        self.config.data_exporter.export_data(data=data_list, filename=filename, filepath=filepath)

    def _make_data_for_export(self):
        data_list=[]
        for sample in self.samples_abstracted_id:
            data = []
            for node in sample:
                node_value = [int(node[0]), node[1]]
                data.append(node_value)
            data_list.append(data)
        return data_list
                

if __name__ == "__main__":
    pass

