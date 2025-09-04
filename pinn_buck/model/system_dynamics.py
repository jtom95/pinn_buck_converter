from abc import ABC, abstractmethod
from typing import Mapping, Dict, List, Tuple, Callable
import torch

from .system_state_class import States
from pinn_buck.parameters.parameter_class import Parameters


class SystemDynamics(ABC):
    """
    Defines the physical system:
      - dynamics(start_states, params) -> next_states
    """

    @abstractmethod
    def dynamics(self, start_states: States, params: Parameters, controls: Dict[str, torch.Tensor] = None, **kwargs) -> States:
        """Return derivatives for the **evolving** states keys"""
        ...
