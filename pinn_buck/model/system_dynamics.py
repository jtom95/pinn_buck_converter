from abc import ABC, abstractmethod
from typing import Mapping, Dict, List, Tuple, Callable
import torch

from .system_state_class import State
from pinn_buck.parameters.parameter_class import Parameters


class SystemDynamics(ABC):
    """
    Defines the physical system:
      - dynamics(current_state, params) -> next_state
    """        

    @abstractmethod
    def dynamics(self, current_state: State, params: Parameters, **kwargs) -> State:
        """Return derivatives for the **evolving** state keys"""
        ...


