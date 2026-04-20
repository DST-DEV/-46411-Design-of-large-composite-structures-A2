from abc import ABC, abstractmethod

import numpy as np

class FailureCriterion(ABC):
    """Failure criterion of a lamina."""
    @abstractmethod
    def failure_index(self, stress: np.ndarray) -> float:
        """Calculate the failure index of a lamina for a given stress state."""
        pass
