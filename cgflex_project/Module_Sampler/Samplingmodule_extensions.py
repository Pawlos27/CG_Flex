
import csv
import pandas as pd
from abc import ABCMeta, abstractstaticmethod, abstractmethod
import os


class IData_exporter(metaclass=ABCMeta):
    @abstractmethod
    def export_data(self, data, filename:str):
     """Interface Method"""



class Csv_exporter_for_id_value_pairs(IData_exporter):
    def __init__(self):
        pass

    def export_data(self, data, filename, filepath=None):
        if filepath == None:
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            filepath = os.path.join(data_folder, filename)
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID","Value"])
            writer.writerows(data)