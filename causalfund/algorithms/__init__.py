"""Domain generalization algorithms for CausalFund."""

from causalfund.algorithms.base import Algorithm
from causalfund.algorithms.erm import ERM
from causalfund.algorithms.casn import CaSN, CaSN_MMD, CaSN_IRM

__all__ = ["Algorithm", "ERM", "CaSN", "CaSN_MMD", "CaSN_IRM"]


def get_algorithm_class(algorithm_name):
    """
    Get algorithm class by name.
    
    Args:
        algorithm_name: Name of the algorithm (e.g., 'ERM', 'CaSN')
    
    Returns:
        Algorithm class
    """
    algorithms = {
        'ERM': ERM,
        'CaSN': CaSN,
        'CaSN_MMD': CaSN_MMD,
        'CaSN_IRM': CaSN_IRM,
    }
    
    if algorithm_name not in algorithms:
        raise NotImplementedError(
            f"Algorithm '{algorithm_name}' not implemented. "
            f"Available: {list(algorithms.keys())}"
        )
    
    return algorithms[algorithm_name]

