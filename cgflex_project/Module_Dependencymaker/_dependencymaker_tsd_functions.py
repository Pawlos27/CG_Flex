from abc import ABCMeta, abstractstaticmethod, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Type, Tuple
from cgflex_project.Shared_Classes.distributions import *






class ITsd_functions(metaclass=ABCMeta):
    """
    Implementations of this interface enable the integration of Time-Series-Data
    functionalities into our framework. Functions dependent on a time value are implemented.
    In our model, this time value is determined by a counter that increases by 1 after each
    sampling round. The implemented functions should be suitable for this integer value
    range.

    """
    @abstractmethod
    def calculate_value(self, x)-> float:
     """ Calculates the function value based on the input value x, here
        the time value.

        Args:
            x: The input value.

        Returns:
            float: The calculated value of the TSD function.
        """
    def set_range(self,range:tuple):
        """
        Sets the output range of the TSD function, to match our ovrall range.

        Args:
            range (tuple): A tuple specifying the minimum and maximum range for the function's output.
        """
        self.range = range
    def plot_function(self, label="tsd_function"):
        """
        Plots the TSD function over a specified time range as input range, its set to 20.

        Args:
            label (str): The label for the plot.
        """

        plot_input_range = range(0, 20)
        plot_output = [self.calculate_value(x=x) for x in plot_input_range]
        
        plt.figure(figsize=(10, 6))
        plt.xticks(range(0, 20))
        plt.plot(plot_input_range, plot_output, label=self.__class__.__name__, marker='o')
        plt.xlabel('Time Value')
        plt.ylabel('Output Value')
        plt.title(label)
        plt.legend()
        plt.grid(True)
        plt.show()

class Tsd_function_sinus(ITsd_functions):
    """
    Sinusoidal TSD function implementation.

    Args:
        stretch_factor (int): Factor to stretch the sinusoidal function.
    """
    def __init__(self, stretch_factor=10):
       self.stretch_factor = stretch_factor
       self.range = (0,1)

   
    def calculate_value(self,x):

        a, b = self.range
        scaled_output = a + ((b - a) / 2) * (np.sin(x / self.stretch_factor) + 1)
        return scaled_output



class Tsd_function_linear_cycle(ITsd_functions):
    """
    Linear cyclic TSD function implementation.

    Args:
        cycle_length (int): The length of the cycle in the linear function, regarding time units default at 20 .
    """
    def __init__(self, cycle_length=20):
       self.cycle_length = cycle_length   
       self.range = (0,1)
    def calculate_value(self,x):
        cycle_length = self.cycle_length
        a, b = self.range
        range_span = b - a
        # Modulo operation to make the value repeat every cycle_length
        cycle_position = x % cycle_length
        # Calculate the linear position in the cycle
        if cycle_position <= cycle_length / 2:
            # First half of the cycle (ascending)
            return a + 2 * range_span * cycle_position / cycle_length
        else:
            # Second half of the cycle (descending)
            return b - 2 * range_span * (cycle_position - cycle_length / 2) / cycle_length


class Tsd_function_custom_list(ITsd_functions):
    """
    Custom list based TSD function implementation. The function output is taken from the provided list, input is used to iterate trough the list.

    Args:
        value_list (list): A list of values to be used in the TSD function.
    """
    def __init__(self, value_list:list):
        self.value_list = value_list   
        self.range = (0,1)
        if not all(0 <= val <= 1 for val in value_list):
            raise ValueError("All values in the list must be between 0 and 1")

    def calculate_value(self, x):
        value_list = self.value_list
        # Cyclically select a value from the list
        selected_value = value_list[x % len(value_list)]

        # Scaling and shifting the selected value to fit the output range
        a, b = self.range
        scaled_output = a + (b - a) * selected_value
        return scaled_output


class ITsd_function_collection(metaclass=ABCMeta):
    """
    Offers the possibility to implement various collections that provide a preselection
    of Time Series Data (TSD) functions.
    """
    @abstractmethod
    def __init__(self):
        """
        Initializes the Tsd_function_collection_full with a list of predefined set of TSD functions.
        """
        self.functions : List[ITsd_functions]
    @abstractmethod
    def get_tsd_function(self)-> ITsd_functions:
     """
        Selects and returns a TSD function from the collection.

        Returns:
            ITsd_functions: A randomly chosen TSD function instance from the collection.
        """
    def set_range(self,range: tuple):
        """
        Sets the output range for all TSD functions in the collection.

        Args:
            range (tuple): A tuple specifying the minimum and maximum range for the functions' output.
        """
        self.range = range


class Tsd_function_collection_full(ITsd_function_collection):
    """
    Concrete implementation of ITsd_function_collection.
    """
    def __init__(self):
        self.functions = [Tsd_function_sinus(),Tsd_function_linear_cycle()]
    def get_tsd_function(self):
       random_tsd_function = random.choice(self.functions)
       random_tsd_function.set_range(range=self.range)
       return random_tsd_function



if __name__ == "__main__":

    #funktion = Tsd_function_linear_cycle(cycle_length=10)
    function = Tsd_function_sinus(stretch_factor=2)
    function.plot_function(label="sinus      stretch factor=2 ")
    pass