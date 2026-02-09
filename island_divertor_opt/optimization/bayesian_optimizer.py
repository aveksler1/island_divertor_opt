import os
import json
import numpy as np
import scipy
import argparse
import pickle
import time
import sys
import subprocess
import itertools

from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

from sklearn.gaussian_process.kernels import Matern

# We must include the script before more python imports, or else we run into
# HDF5 errors.
from island_divertor_opt import nersc_bridge as jb
script_path = "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/make_divertor_module.jl"
jb.jl.include(script_path)

from island_divertor_opt.optimization.flare_postprocessing import get_cost_vars, calculate_cost

import matplotlib.pyplot as plt

class BayesianOptimizer(object):
    """
    This class performs a Bayesian optimization using an automatic divertor
    making algorithm.


    """
    def __init__(self, 
        root_dir, 
        init_points, 
        n_iter, 
        min_bound_left, 
        max_bound_left, 
        min_bound_right, 
        max_bound_right, 
        domain_lower_prefac,
        domain_upper_prefac,
        xi=0.05,
        nu=2.5,
        random_state=None
    ):
        self.root_dir = root_dir
        self.init_points = init_points
        self.n_iter = n_iter
        self.min_bound_left = min_bound_left
        self.max_bound_left = max_bound_left
        self.min_bound_right = min_bound_right
        self.max_bound_right = max_bound_right
        self.domain_lower_prefac = domain_lower_prefac
        self.domain_upper_prefac = domain_upper_prefac
        self.xi = xi
        self.nu = nu
        self.random_state = random_state
        
        self.iteration = 0
        self.CAP_COST = 1

        # 1. 
        os.makedirs(self.root_dir, exist_ok=True)
        self._save_settings()
        self._print_settings()

    def _save_settings(self):
        """Saves all init parameters to a JSON file for future reference."""
        settings = {
            "init_points": self.init_points,
            "n_iter": self.n_iter,
            "bounds_left": [self.min_bound_left, self.max_bound_left],
            "bounds_right": [self.min_bound_right, self.max_bound_right],
            "kernel_constraints": {
                "lower_prefac": self.domain_lower_prefac,
                "upper_prefac": self.domain_upper_prefac
            },
            "random_state": self.random_state
        }
        filepath = os.path.join(self.root_dir, "settings.json")
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=4)

    def _print_settings(self):
        """Prints a clear summary to the console when the simulation starts."""
        print("-" * 30, flush=True)
        print("SIMULATION INITIALIZED", flush=True)
        print(f"Directory: {self.root_dir}", flush=True)
        print(f"Points:    {self.init_points} init, {self.n_iter} iter", flush=True)
        print(f"THETA_L Range:  [{self.min_bound_left}, {self.max_bound_left}]", flush=True)
        print(f"THETA_R Range:  [{self.min_bound_right}, {self.max_bound_right}]", flush=True)
        print(f"Kernel:    Matern (nu=2.5)", flush=True)
        print(f"LS Bounds: {self.domain_lower_prefac}x to {self.domain_upper_prefac}x domain", flush=True)
        print("-" * 30, flush=True)

    def initialize_optimization(self):

        self.theta_1s = np.linspace(self.min_bound_left, self.max_bound_left, 1000)
        self.theta_2s = np.linspace(self.min_bound_right, self.max_bound_right, 1000)

        self.acquisition_function = acquisition.ExpectedImprovement(xi=self.xi)

        pbounds = {
            'x': (self.min_bound_left, self.max_bound_left),
            'y': (self.min_bound_right, self.max_bound_right)
        }

        self.optimizer = BayesianOptimization(
            f=self.cost,
            pbounds=pbounds,
            acquisition_function=self.acquisition_function,
            random_state=self.random_state
        )

        domain_range = np.max(
            [
                self.max_bound_left - self.min_bound_left, 
                self.max_bound_right-self.min_bound_right
            ]
        )
        lower = domain_range * self.domain_lower_prefac
        upper = domain_range * self.domain_upper_prefac
        bounds = (lower, upper)
        print(f"Constraining length scale to {bounds}", flush=True)

        kernel = Matern(
            length_scale=domain_range * 0.5, 
            length_scale_bounds=bounds, 
            nu=self.nu
        )

        self.optimizer.set_gp_params(kernel=kernel, normalize_y=True, alpha=1e-2)
            
    def run_optimization(self):
        self.optimizer.maximize(init_points=self.init_points, n_iter=0)

        for _ in range(self.n_iter):
            self.optimizer.maximize(init_points=0, n_iter=1)
            self.plot_gp()       
            self.optimizer.save_state(f"optimizer_state.json")

        self.optimizer.save_state(os.path.join(f"{self.root_dir}", "final_optimizer_state.json"))

    def run_simulation(self, left_pt, right_pt, dirname):
        """
        From input parameters (left starting point, right starting point
        strike angle), builds a divertor and runs flare to evaluate divertor
        performance.
        """
        print("Setting up environment...", flush=True)
        # Change back to root directory after every time we run the simulation
        os.chdir(self.root_dir)
        os.makedirs(dirname, exist_ok=True)
        os.chdir(dirname)
        os.makedirs("boundary", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        # Copy json files over
        # subprocess.run(
        #     ["cp", "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/classical_divertor_fwd.json", "."],
        #     capture_output=True,
        #     check=True    
        # )
        # subprocess.run(
        #     ["cp", "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/classical_divertor_bwd.json", "."],
        #     capture_output=True,
        #     check=True
        # )
        # Make the actual divertor given the left point and right point
        print("Making divertor...", flush=True)
        t = time.time()
        jb.jl.DivertorMaker.run_divertor_creation(left_pt, right_pt)
        print(f"Time to make divertor: {time.time() - t} s", flush=True)
        print("Moving files...", flush=True)

        # Move the divertor into the boundary folder
        subprocess.run(
            ["mv", "classical_fwd.txt", "classical_bwd.txt", "classical_fwd_cap.txt", "classical_bwd_cap.txt", "boundary"],
            capture_output=True,
            check=True    
        )
        subprocess.run(
            ["cp", "/global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/boundary/wall.txt", "boundary"],
            capture_output=True,
            check=True        
        )

        # Enter the output folder
        os.chdir("output")

        # Run flare in the output folder
        print("Running flare...", flush=True)
        t = time.time()
        subprocess.run(
            f"mpirun -np 128 python /global/homes/a/aveksler/t1e/repositories/island_divertor_opt/island_divertor_opt/scripts/run_flare.py {os.getcwd()}", 
            shell=True,
            capture_output=True,
            check=True        
        )
        print(f"Time to run FLARE: {time.time() - t} s", flush=True)

        # Perform postprocessing to get cost variables
        print(f"Running postprocessing on dir {dirname}...", flush=True)
        qpeak, cap_hits = get_cost_vars(os.path.join(f"{self.root_dir}/{dirname}/output"))
        return qpeak, cap_hits

    def cost(self, x, y):
        lp = x
        rp = y
        print(f"Running sim: iter {self.iteration}; (left point, right point) = ({lp}, {rp}))", flush=True)
        self.iteration+=1

        # Determine the phase and the specific ID for that phase
        if self.iteration <= self.init_points:
            phase = "initial"
            idx = self.iteration
        else:
            phase = "iteration"
            idx = self.iteration - self.init_points
            
        # Create a zero-padded directory name (e.g., "initial_05" or "iter_012")
        dirname = f"{phase}_{idx:03d}"

        try:
            qpeak, cap_hits = self.run_simulation(lp, rp, dirname)
            money = calculate_cost(qpeak, cap_hits, CAP_COST=self.CAP_COST)
            # Save JSON with data
            data_to_save = {
                "iter": self.iteration,
                "left_point": lp,
                "right_point": rp,
                "cost": money,
                "qpeak": qpeak,
                "cap%": cap_hits
            }
            with open("data.json", "w") as io:
                json.dump(data_to_save, io, indent=4)
            # Increase iteration count
            # Make negative so maximizing will actually minimize
            return -money
        except Exception as e:
            print(f"Iteration {self.iteration} failed with error: {e}", flush=True)
            return -200
        # lp = x
        # rp = y
        # print(f"Running sim: iter {self.iteration}; (left point, right point) = ({lp}, {rp}))")
        # qpeak, cap_hits = self.run_simulation(lp, rp)
        # money = calculate_cost(qpeak, cap_hits, CAP_COST=self.CAP_COST)
        # # Save JSON with data
        # data_to_save = {
        #     "iter": self.iteration,
        #     "left_point": lp,
        #     "right_point": rp,
        #     "cost": money,
        #     "qpeak": qpeak,
        #     "cap%": cap_hits
        # }
        # with open("data.json", "w") as io:
        #     json.dump(data_to_save, io, indent=4)
        # # Increase iteration count
        # # Make negative so maximizing will actually minimize
        # self.iteration+=1
        # return -money

    def plot_gp(self):
        all_combinations = list(itertools.product(self.theta_1s, self.theta_2s))
        grid = np.array(all_combinations)

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        steps = len(self.optimizer.space)
        fig.suptitle(
            f"Bayesian optimization after {steps - self.init_points} steps"
        )

        theta_1s_obs = np.array([[res["params"]["x"] for res in self.optimizer.res]])
        theta_2s_obs = np.array([[res["params"]["y"] for res in self.optimizer.res]])
        target_obs = np.array([res["target"] for res in self.optimizer.res])

        # self.optimizer.set_gp_params(normalize_y=True, alpha=1e-2)
        self.optimizer._gp.fit(self.optimizer.space.params, self.optimizer.space.target)
        self.optimizer.acquisition_function._fit_gp(self.optimizer._gp, self.optimizer._space)
        mu, sigma = self.optimizer._gp.predict(grid, return_std=True)

        # Plot mean value
        len_theta = len(self.theta_1s)
        im0 = axes[0].pcolormesh(self.theta_1s, self.theta_2s, mu.reshape(len_theta, len_theta).T)
        plt.colorbar(im0, ax=axes[0])
        axes[0].scatter(theta_1s_obs, theta_2s_obs, label=u'Observations', color='r')

        # Plot utility function
        utility_function = self.acquisition_function
        utility_function.y_max = self.optimizer._space._target_max()
        utility = -1 * utility_function._get_acq(gp=self.optimizer._gp)(grid)
        
        im1 = axes[1].pcolormesh(self.theta_1s, self.theta_2s, utility.reshape(len_theta, len_theta).T, vmin=np.min(utility), vmax=np.max(utility))
        plt.colorbar(im1, ax=axes[1])
        axes[1].scatter(theta_1s_obs, theta_2s_obs, label=u'Observations', color='r')
        flattened_argmax = np.argmax(utility)
        row, col = np.unravel_index(flattened_argmax, utility.reshape(len_theta, len_theta).shape)
        axes[1].scatter(self.theta_1s[row], self.theta_2s[col], marker='*', color='k', label="Next Best Guess")
        print(f"Next best guess at ({self.theta_1s[row]}, {self.theta_2s[col]})")
        
        axes[0].set_title("Gaussian Process Mean")
        axes[1].set_title("Acquisition function")

        for ax in axes:
            ax.set_xlabel("$\\theta_l$")
            ax.set_ylabel("$\\theta_r$")
            ax.legend()

        plt.savefig(f"gp_iter_{self.iteration}.png", dpi=600)
        plt.savefig(f"gp_iter_{self.iteration}.pdf")

        plt.show()
