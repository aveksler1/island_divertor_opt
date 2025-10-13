import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from moose.data import Dataset

import vtk
import vtkmodules.vtkFiltersGeometry as vtkFiltersGeometry

class HeatFluxDiagnostic(object):
    """
    Determines hit percentage on different divertor surfaces.

    """
    def __init__(self, project_path, power=8, verbose=False):
        self.project_path = project_path
        # self.
        try:
            self.hflux = Dataset.loadnc(os.path.join(project_path, "heat_flux.nc"))
        except:
            print("No heat flux diagnostic")
        try: 
            self.strike_points = Dataset.loadnc(os.path.join(project_path, "strike_points.nc"))
        except:
            print("No strike point diagnostic")
        self.verbose = verbose
        self.power = power # MW

    def calculate_peak_hflux(self):
        if not hasattr(self, 'hflux'):
            raise Exception("hflux no provided! Cannot return peak hflux.")

        data = self.hflux['hload']*self.power
        self.peak_hflux = 0
        for i in range(0, len(data.values)):
            if np.max(data.values[i]) > self.peak_hflux:
                self.peak_hflux = np.max(data.values[i])
        if self.verbose:
            print(f"Peak Heat Flux: {self.peak_hflux}")

    def calculate_total_heat(self):
        """
        Assumes the divertor surfaces are surfaces 1 and 2
        0 - wall
        1 - fwd
        2 - bwd
        3 - cap fwd
        4 - cap bwd
        """
        self.total_q = 0
        for i in [1, 2]:
            hflux = (self.hflux['hload']*self.power).values[i]
            area = self.hflux['area'].values[i]
            heat = hflux * area
            self.total_q += np.sum(heat)
            if self.verbose:
                print(f"Area of plate {i}: {np.sum(area)}")
                print(f"Heat on plate {i}: {np.sum(heat)}")

        return self.total_q

    # def calculate_area(self, data, boundary_idx):
    #     """
    #     Calculates the area of a boundary surface given the boundary index and
    #     the object to source the grid from
    #     """

    #     # grid = self.hflux['hload'].grid.xcart[boundary_idx]
    #     grid = data.grid.xcart[boundary_idx]

    #     def create_vector(p0, p1):
    #         """Creates a vector point to p1 from p0"""
    #         return np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]])

    #     area_matrix = np.zeros((grid.shape[0] - 1, grid.shape[1] - 1))

    #     total_area = 0

    #     for i in range(grid.shape[0] - 1):
    #         for j in range(grid.shape[1] - 1):
    #             p1 = grid[i, j]
    #             p2 = grid[i, j+1]
    #             p3 = grid[i+1, j]
    #             p4 = grid[i+1, j+1]

    #             # Create first triangle
    #             v13 = create_vector(p1, p3)
    #             v12 = create_vector(p1, p2)
    #             a1 = np.linalg.norm(np.cross(v13, v12))/2

    #             # Create second triangle
    #             v43 = create_vector(p4, p3)
    #             v42 = create_vector(p4, p2)
    #             a2 = np.linalg.norm(np.cross(v43, v42))/2
                
    #             total_area += (a1+a2)
    #             area_matrix[i, j] = (a1+a2)

    #     return area_matrix

    def calculate_evenness(self):
        data = self.strike_points['p']
        self.evenness_dict = {}
        for i in range(0, len(data.values)):
            self.evenness_dict['std_dev_tor'] = np.sum(np.std(data.values[i], axis=0))
            self.evenness_dict['std_dev_pol'] = np.sum(np.std(data.values[i], axis=1))
            if self.verbose:
                print(f"Summed standard deviation toroidal: {self.evenness_dict['std_dev_tor']}")
                print(f"Summed standard deviation poloidal: {self.evenness_dict['std_dev_pol']}")

    def calculate_percentage(self):
        data = self.strike_points['p']
        self.hit_percentage_dict = {}
        total = 0
        for i in range(0, len(data.values)):
            total += np.nansum(data.values[i])
        for i in range(0, len(data.values)):
            percentage = 100*np.nansum(data.values[i])/total
            self.hit_percentage_dict[i] = percentage
            if self.verbose:
                print(f"Hit % on surface {i}: {percentage:.3f} %")

    def distance(self, p1, p2):
        return np.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2 )

    def reformat_data(self, grid, data, area):
        corners = np.zeros(4, dtype=int)
        ccount = 0
        for j in range(grid.shape[0]-1):
            p_cur = grid[j, 0, :] # extract at 0th index because it doesn't matter
            p_next = grid[j+1, 0, :]
            if j == 0:
                nprev = (p_next - p_cur) / np.linalg.norm(p_next - p_cur)
                ncur = nprev
            else:
                nprev = ncur
                ncur = (p_next - p_cur) / np.linalg.norm(p_next - p_cur)
                if not np.allclose(ncur, nprev):
                    corners[ccount] = j
                    ccount+=1
        corners[-1] =  grid.shape[0]

        # Extract the range where all the data lives
        ranges = [
            (0, corners[0]), 
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3])    
        ]

        best_range = np.argmax(
            [
                np.max(data[ranges[0][0]:ranges[0][1]]),
                np.max(data[ranges[1][0]:ranges[1][1]]),
                np.max(data[ranges[2][0]:ranges[2][1]]),
                np.max(data[ranges[3][0]:ranges[3][1]]),
            ]
        )

        # Once again cull data and grid to only contain the edge with data
        data = data[ranges[best_range][0]:ranges[best_range][1] - 1] # We don't want to grab extra data!
        area = area[ranges[best_range][0]:ranges[best_range][1] - 1]
        grid = grid[ranges[best_range][0]:ranges[best_range][1]]

        heat = area*data

        # Bin based on distance from mean positions. Make sure its aligned to 
        # data grid instead of grid 
        mean_pos = np.mean((grid[:, 1:, :] + grid[:, :-1, :])/2, axis=0)

        dist_to_center = np.empty(shape = data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                p1 = grid[i, j]
                p2 = grid[i+1, j]
                p3 = grid[i, j+1]
                p4 = grid[i+1, j+1]
                data_pos = np.mean(np.array([p1, p2, p3, p4]), axis=0)
                d = self.distance(data_pos, mean_pos[j])
                if i < data.shape[0]/2: # negative distances on one side of divertor plate
                    d*=-1
                dist_to_center[i, j] = d


        # Extract edges to bin
        edge = np.max(dist_to_center)
        nbins = 70
        bins = np.linspace(-edge, edge, nbins)
        output = np.full((len(bins) - 1, data.shape[1]), np.nan)

        # Bin distance to center in new bins
        for i in range(data.shape[1]):
            hist, _ = np.histogram(dist_to_center[:, i], bins=bins, weights=heat[:, i], density=False)
            hist_area, _ = np.histogram(dist_to_center[:, i], bins=bins, weights=area[:, i], density=False)
            output[:, i] = hist / hist_area 
            
        return output, bins

    def plot(self):
        grid_fwd = self.hflux['hload'].grid.xcart[1]
        data_fwd = self.hflux['hload'].values[1]
        area_fwd = self.hflux['area'].values[1]
        grid_bwd = self.hflux['hload'].grid.xcart[2]
        data_bwd = self.hflux['hload'].values[2]
        area_bwd = self.hflux['area'].values[2]

        output_bwd, _ = self.reformat_data(grid_bwd, data_bwd, area_bwd)
        output_fwd, bins = self.reformat_data(grid_fwd, data_fwd, area_fwd)

        output = np.hstack((output_bwd, output_fwd))
        
        cmap = cm.magma
        cmap.set_bad('white')

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        xax = np.linspace(15, 45, output_fwd.shape[1] + output_bwd.shape[1] + 1)

        im0 = ax[0].pcolormesh(bins, xax, output.T * self.power, cmap=cmap, shading='flat')
        im1 = ax[1].pcolormesh((bins[1:] + bins[:-1])/2, (xax[1:] + xax[:-1])/2, output.T * self.power, cmap=cmap, shading='gouraud')

        plt.colorbar(im0, ax=ax[0], label = "Heat Flux [$\\frac{MW}{m^2}$]")
        ax[0].set_xlabel("distance to center (m)")
        ax[0].set_ylabel("$\\phi$ (deg.)")

        plt.colorbar(im0, ax=ax[1], label = "Heat Flux [$\\frac{MW}{m^2}$]")
        ax[1].set_xlabel("distance to center (m)")
        ax[1].set_ylabel("$\\phi$ (deg.)")

        ax[0].set_title("Shading = flat")
        ax[1].set_title("Shading = gouraud")

        fig.tight_layout()
