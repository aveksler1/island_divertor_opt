import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams.update({
    "font.serif": ["Computer Modern Roman"],  # LaTeX default serif font
    "font.size": 16,
    "legend.fontsize": 10
})


class FilePlotter(object):
    """
    This class plots the divertor from a .txt file
    """
    def __init__(self, project_path, file_name):
        self.project_path = project_path
        self.file_name = file_name
        
    def read_text(self):
        with open(os.path.join(self.project_path, self.file_name)) as f:
            text = f.read()
            text = text.strip()
            entries = text.split('\n')
        data = entries[1].split(' ')
        self.ntoroidal = int(data[0])
        self.npoloidal = int(data[1])
        self.idontremember = int(data[2])
        # Remove first two headers
        entries = entries[2:]
        # Be aware that this introduces minor round off error. i.e 38.50000000000002 -> 38.5
        self.deg_arr = np.round(np.array(entries[::self.npoloidal+1], dtype=np.float32), decimals=2)
        self.wall_dict = {}
        # For every degree, save its r and z values
        for i, deg in enumerate(self.deg_arr):
            # index where degrees are stored. + 1 to account for 
            # data starting at index 1
            idx = i*(self.npoloidal + 1) + 1
            data = entries[idx:idx+self.npoloidal]
            # Grab first and second entry (data is stored as r, z)
            r = np.array([j.split()[0] for j in data], dtype=np.double) / 100
            z = np.array([j.split()[1] for j in data], dtype=np.double) / 100
            self.wall_dict[deg] = {'r':r, 'z':z}

        self.min_phi = self.deg_arr[0]
        self.max_phi = self.deg_arr[-1]

    def get_interp_pts(self, phi):
        """
        Get data corresponding to phi 
        """
        if phi in self.deg_arr:
            print(f"Phi={phi} found in dict.")
            return self.wall_dict[phi]['r'], self.wall_dict[phi]['z']

        if phi < self.min_phi or phi > self.max_phi:
            raise Exception(f"Requested phi = {phi} out of range of phi = ({self.min_phi}, {self.max_phi}).")
        
        idx = np.searchsorted(self.deg_arr, phi)
        
        fdist = self.deg_arr[idx] - phi
        bdist = phi - self.deg_arr[idx - 1]

        phi_requested = phi
        fphi = self.deg_arr[idx]
        bphi = self.deg_arr[idx-1]
        
        print(f"(phi requested, forward phi, backwards phi) = ({phi_requested},{fphi},{bphi})", flush=True)
        print(f"(fdist, bdist) = ({fdist}, {bdist})")
        
        r = self.wall_dict[self.deg_arr[idx]]['r'] * (bdist / (fdist + bdist)) + self.wall_dict[self.deg_arr[idx -1]]['r'] * (fdist / (fdist + bdist))
        z = self.wall_dict[self.deg_arr[idx]]['z'] * (bdist / (fdist + bdist)) + self.wall_dict[self.deg_arr[idx -1]]['z'] * (fdist / (fdist + bdist))

        return r, z
        
    def plot_slice(self, phi):
        fig, ax = plt.subplots(1, 1)
        r, z = self.get_interp_pts(phi)
        ax.plot(r, z)
        ax.set_xlabel('r (cm)')
        ax.set_ylabel('z (cm)')
        ax.set_title(f"Slice at phi={phi} degrees")

        return ax

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("project_path", 
    help="Folder where the input file lives",
    type=str
)
parser.add_argument("--wall_file", 
    help="Wall file name",
    type=str,
    default="wall.txt"
)
parser.add_argument("--divertor_file", 
    help="Divertor file name",
    type=str,
    default="classical_fwd.txt"
)
parser.add_argument("--island_boundary_file", 
    help="Island boundary file name",
    type=str,
    default="/global/cfs/cdirs/m4680/av3249/research/divertor_paper/data/input_files/island_boundary_halfdegree.csv"
)
parser.add_argument("--pointsperperiod",
    help="How many points to go around",
    type=int,
    default=180
)
parser.add_argument("--phi",
    help="Which phi to plot (for island boundary, wall, divertor)",
    type=float,
    default=0.
)

# Read in arguments
args = parser.parse_args()
project_path = args.project_path
island_boundary_file = args.island_boundary_file
wall_file = args.wall_file
divertor_file = args.divertor_file
pointsperperiod = args.pointsperperiod
phi = args.phi

print("==== INPUT ARGS ====")
print(f"Project path: {project_path}")
print(f"Boundary file: {island_boundary_file}")
print(f"Wall file: {wall_file}")
print(f"Divertor file: {divertor_file}")
print(f"Points per period: {pointsperperiod}")
print(f"Phi: {phi}")

# Plotting Routine
fig, ax = plt.subplots(1, 1)

# Plotting Wall
if wall_file is not None:
    wall = FilePlotter(project_path, wall_file)
    wall.read_text()
    r, z = wall.get_interp_pts(phi)
    ax.plot(r, z, label = 'Wall')
else:
    print("Wall file not provided.")
# Plotting boundary
if island_boundary_file is not None:
    # Plotting the island boundary expects an int, so we round to
    # the nearest int 
    startval = round(phi*2)
    r = []
    z = []
    print("startval: ", startval)
    with open(island_boundary_file) as wf:
        npoints = 0 # total lines divided by number of islands
        for count, line in enumerate(wf):
            npoints += 1
        npoints = int(npoints / pointsperperiod / 5)
        print("NPOINTS: ", npoints)
    with open(island_boundary_file) as wf:
        for count, line in enumerate(wf):
            if (count - startval) % pointsperperiod == 0:
                dum = line.split(",")
                # convert to cm
                r.append(float(dum[0]))
                z.append(float(dum[1]))

    # ax.scatter(r[::5], z[::5], color='r', label= 'Control Surface')
    # Plot each island by shifting the starting index (0 through 4) and stepping by 5
    ax.scatter(r[0::5], z[0::5], color='r', label='Cross section 1')
    ax.scatter(r[1::5], z[1::5], color='g', label='Cross section 2')
    ax.scatter(r[2::5], z[2::5], color='b', label='Cross section 3')
    ax.scatter(r[3::5], z[3::5], color='y', label='Cross section 4')
    ax.scatter(r[4::5], z[4::5], color='c', label='Cross section 5')

    # ax.scatter(r[:npoints], z[:npoints], color='r', label= 'Control Surface 1')
    # ax.scatter(r[npoints:2*npoints], z[npoints:2*npoints], color='g', label= 'Control Surface 2')
    # ax.scatter(r[2*npoints:3*npoints], z[2*npoints:3*npoints], color='b', label= 'Control Surface 3')
    # ax.scatter(r[3*npoints:4*npoints], z[3*npoints:4*npoints], color='y', label= 'Control Surface 4')
    # ax.scatter(r[4*npoints:], z[4*npoints:], color='c', label= 'Control Surface 5')
else:
    print("Boundary file not provided.")
# Plotting divertor
# if divertor_file is not None:
#     div = FilePlotter(project_path, divertor_file)
#     div.read_text()
#     r, z = div.get_interp_pts(phi)
#     ax.scatter(r, z, marker='*')
#     ax.plot(r, z, label = "Divertor")
# else:
#     print("Divertor file not provided.")

ax.set_xlim([9, 14.1])
ax.set_title(f'$\phi$={phi} degrees')
ax.legend(loc='lower left')
ax.set_xlabel('r [m]')
ax.set_ylabel('z [m]')

fig.tight_layout()

plt.savefig(f"cross_section_phi={phi}.pdf")
