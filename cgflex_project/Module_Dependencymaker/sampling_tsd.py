
from typing import Any, List, Type, Tuple, Optional
from abc import ABCMeta, abstractstaticmethod, abstractmethod

class ITsd_strategy_setter(metaclass=ABCMeta):
    @abstractmethod
    def set_sources(self):
        """Interface Method"""


class Tsd_manual(ITsd_strategy_setter):
    def __init__(self):
        pass
    def set_sources(self):
        pass

class tsd_function(ITsd_strategy_setter):
    def __init__(self):
        pass
    def set_sources(self):
        pass

if __name__ == "__main__":
    pass