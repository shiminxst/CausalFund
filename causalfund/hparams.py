"""
Hyperparameter registry for CausalFund algorithms.

Provides default hyperparameters for each algorithm and dataset.
"""

import numpy as np


def default_hparams(algorithm, dataset='FundusGlaucoma'):
    """
    Get default hyperparameters for an algorithm.
    
    Args:
        algorithm: Algorithm name (e.g., 'ERM', 'CaSN')
        dataset: Dataset name (currently only 'FundusGlaucoma' supported)
    
    Returns:
        Dictionary of hyperparameters
    """
    # Base hyperparameters (common to all algorithms)
    hparams = {}
    
    # Data augmentation
    hparams['data_augmentation'] = True
    
    # Model architecture
    hparams['model_arch'] = 'resnet50'
    hparams['resnet_dropout'] = 0.0
    hparams['nonlinear_classifier'] = False
    hparams['pretrained'] = True
    
    # Optimization
    hparams['lr'] = 5e-5
    hparams['weight_decay'] = 0.0
    hparams['batch_size'] = 32
    
    # Algorithm-specific hyperparameters
    if algorithm == 'ERM':
        # Empirical Risk Minimization (no additional params)
        pass
    
    elif algorithm in {'CaSN', 'CaSN_MMD', 'CaSN_IRM'}:
        # CaSN-specific hyperparameters
        hparams['bias'] = 3.0              # Intervention strength
        hparams['int_lambda'] = 1.0        # Intervention loss weight
        hparams['kl_lambda'] = 0.01        # KL divergence weight
        hparams['int_reg'] = 0.1           # Intervention regularization
        hparams['target_lambda'] = 0.1     # Target consistency weight
        hparams['prior_type'] = 'conditional'  # Type of prior ('conditional' or 'standard')
        hparams['max_optimization_step'] = 1  # Steps for adversarial training
        hparams['if_adversarial'] = False  # Whether to use adversarial training
        if algorithm == 'CaSN_MMD':
            hparams['mmd_weight'] = 1.0
            hparams['mmd_kernel'] = 'gaussian'
            hparams['mmd_gamma'] = [0.5, 1.0, 2.0]
        if algorithm == 'CaSN_IRM':
            hparams['irm_lambda'] = 1000.0
            hparams['irm_penalty_anneal_iters'] = 500
    
    else:
        raise NotImplementedError(f"No hyperparameters defined for algorithm '{algorithm}'")
    
    return hparams


def random_hparams(algorithm, dataset='FundusGlaucoma', seed=0):
    """
    Sample random hyperparameters for an algorithm (for hyperparameter search).
    
    Args:
        algorithm: Algorithm name
        dataset: Dataset name
        seed: Random seed
    
    Returns:
        Dictionary of randomly sampled hyperparameters
    """
    # Start with defaults
    hparams = default_hparams(algorithm, dataset)
    
    # Set random seed
    rng = np.random.RandomState(seed)
    
    # Randomly sample some key hyperparameters
    hparams['lr'] = 10 ** rng.uniform(-5, -3.5)  # 1e-5 to ~3e-4
    hparams['weight_decay'] = 10 ** rng.uniform(-6, -2)  # 1e-6 to 1e-2
    hparams['batch_size'] = int(2 ** rng.uniform(4, 6))  # 16 to 64
    
    if algorithm in {'CaSN', 'CaSN_MMD', 'CaSN_IRM'}:
        hparams['bias'] = rng.uniform(1.0, 5.0)
        hparams['int_lambda'] = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
        hparams['kl_lambda'] = 10 ** rng.uniform(-3, -1)  # 0.001 to 0.1
        if algorithm == 'CaSN_MMD':
            hparams['mmd_weight'] = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
            # Sample gamma list by scaling base values
            base_gamma = np.array([0.5, 1.0, 2.0])
            scale = 10 ** rng.uniform(-1, 1)
            hparams['mmd_gamma'] = (base_gamma * scale).tolist()
        if algorithm == 'CaSN_IRM':
            hparams['irm_lambda'] = 10 ** rng.uniform(2, 4)  # 100 to 10_000
            hparams['irm_penalty_anneal_iters'] = int(rng.uniform(100, 1000))
    
    return hparams

