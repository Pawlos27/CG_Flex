from dataclasses import dataclass, field
from typing import Any, List, Type, Union




@dataclass(order=True)
class Nodeobject:
    """
    Data class representing a node object with attributes used to represent graphs and dependencies in all stages of the framework.
    This class is crucial for representing individual nodes. Its dependencies are gathered in a list to represent graph structures and can be populated with dependencies for use in later stages of data sampling.

    """
    sort_index: float = field(init=False, repr=False)
    """A derived value used for sorting nodes, typically based on 'coord_l'."""
    id : int =field(default_factory=int)
    """A unique identifier for the node, related to the coord_l value for which it is assigned in ascending order."""
    coord_t : list = field(default_factory=list)
    """Coordinates in the 'thorus' dimensions, representing the node's position, is a list because the thorus can have arbitrary dimensions ."""
    coord_l : float = field(default_factory=float)
    """Linear coordinate, used for layer-based positioning in the network."""
    layer : int = field(default_factory=int)
    """Represents the layer or level in which the node resides within the network and along the coord_L."""
    sink : bool = False
    """Marks the node as a 'sink' in the network topology."""
    source : bool = False
    """Marks the node as a 'source' or origin point within the network."""
    value : float = field(default_factory=float)
    """A continuous value associated with the node, used for sampling data."""
    dependency = None
    """An attribute to represent a node's dependency on other elements, can also be a value or a distribution."""
    parents: list = field(default_factory=list)
    """A list of parent nodes, denoting the ids of the node's immediate predecessors."""
    children: list = field(default_factory=list)
    """A list of child nodes, indicating the ids of the node's immediate successors."""
    scale_val: float = field(default_factory=float)
    """A scaling factor, representing by which factor the space the node lies in is scaled."""

    def __post_init__(self):    
        self.sort_index = self.coord_l