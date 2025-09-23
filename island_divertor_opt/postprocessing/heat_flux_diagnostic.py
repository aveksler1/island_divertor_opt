import os
import sys
import numpy as np
from moose.data import Dataset

import vtk
import vtkmodules.vtkFiltersGeometry as vtkFiltersGeometry

class HeatFluxDiagnostic(object):
    """
    Determines hit percentage on different divertor surfaces.

    Mode 0: Use heat flux diagnostic
    Mode 1: Use strike point diagnostic
    """
    def __init__(self, project_path, verbose=False):
        self.project_path = project_path
        try:
            self.hflux = Dataset.loadnc(os.path.join(project_path, "heat_flux.nc"))
        except:
            print("No heat flux diagnostic")
        try: 
            self.strike_points = Dataset.loadnc(os.path.join(project_path, "strike_points.nc"))
        except:
            print("No strike point diagnostic")
        else:
            raise Exception("Mode must be either 0 or 1!")
        self.verbose = verbose
        self.power = 2 # MW

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

    def calculate_total_heat(self, data):
        """
        Assumes the divertor surfaces are surfaces 1 and 2
        0 - wall
        1 - fwd
        2 - bwd
        3 - cap fwd
        4 - cap bwd
        """
        # data = self.hflux['hload']
        self.total_q = 0
        for i in [0, 1, 2, 3, 4]:
            hflux = data.values[i]
            area = self.calculate_area(data, i)
            heat = hflux * area
            self.total_q += np.sum(heat)
            if self.verbose:
                print(f"Area of plate {i}: {np.sum(area)}")
                print(f"Heat on plate {i}: {np.sum(heat)}")

        return self.total_q

    def calculate_area(self, data, boundary_idx):
        """
        Calculates the area of a boundary surface given the boundary index and
        the object to source the grid from
        """

        # grid = self.hflux['hload'].grid.xcart[boundary_idx]
        grid = data.grid.xcart[boundary_idx]

        def create_vector(p0, p1):
            """Creates a vector point to p1 from p0"""
            return np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]])

        area_matrix = np.zeros((grid.shape[0] - 1, grid.shape[1] - 1))

        total_area = 0

        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                p1 = grid[i, j]
                p2 = grid[i, j+1]
                p3 = grid[i+1, j]
                p4 = grid[i+1, j+1]

                # Create first triangle
                v13 = create_vector(p1, p3)
                v12 = create_vector(p1, p2)
                a1 = np.linalg.norm(np.cross(v13, v12))/2

                # Create second triangle
                v43 = create_vector(p4, p3)
                v42 = create_vector(p4, p2)
                a2 = np.linalg.norm(np.cross(v43, v42))/2
                
                total_area += (a1+a2)
                area_matrix[i, j] = (a1+a2)

        return area_matrix

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