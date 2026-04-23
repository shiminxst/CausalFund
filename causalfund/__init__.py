"""
CausalFund: Domain-Invariant Fundus Image Analysis

A framework for learning robust glaucoma detection models that generalize
from hospital-grade to smartphone-based fundus images using causal 
representation learning.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@institution.edu"

from causalfund.datasets import FundusDataset, FundusDataModule
from causalfund.models import get_model
from causalfund.utils import set_seed, load_config
from causalfund.algorithms import get_algorithm_class
from causalfund.dataloaders import InfiniteDataLoader, FastDataLoader

__all__ = [
    "FundusDataset",
    "FundusDataModule",
    "get_model",
    "set_seed",
    "load_config",
    "get_algorithm_class",
    "InfiniteDataLoader",
    "FastDataLoader",
]

