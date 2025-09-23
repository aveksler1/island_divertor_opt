from flare.mmesh.unstructured import Boundary
import sys

path = sys.argv[1]

Boundary.import_flare(path).savenc("boundary.nc")