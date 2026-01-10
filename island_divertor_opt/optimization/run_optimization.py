"""
Avigdor Veksler, 2025

This script runs the BayesianOptimization for divertor shape optimization. It
interfaces with bayesian_optimizer.py, which holds the logic for setting up and
launching the optimization.
"""

import os
from island_divertor_opt.optimization.bayesian_optimizer import BayesianOptimizer

n_iter = 50
init_points = 10
min_bound_left = 1
min_bound_right = 1
max_bound_left = 4.5
max_bound_right = 4.5
domain_lower_prefac = 0.1
domain_upper_prefac = 1.5

root_dir = os.getcwd()

BO = BayesianOptimizer(
    root_dir,
    init_points, 
    n_iter, 
    min_bound_left, 
    max_bound_left, 
    min_bound_right, 
    max_bound_right, 
    domain_lower_prefac,
    domain_upper_prefac,
)

BO.initialize_optimization()
BO.run_optimization()