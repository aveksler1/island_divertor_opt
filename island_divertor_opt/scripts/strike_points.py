from flare.mmesh import tasks
import sys

path = sys.argv[1]

tasks.init(
    path, 
    "boundary.nc", 
    "tracing.nc")
tasks.strike_point_density(1e-6, 100000, output="strike_points.nc")