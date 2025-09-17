import numpy as np
from heat_flux_diagnostic import HeatFluxDiagnostic

def get_cost_vars()
    diag = HeatFluxDiagnostic(".", verbose=False)
    diag.calculate_peak_hflux()
    diag.calculate_percentage()
    q = diag.peak_hflux
    cap_hits = diag.hit_percentage_dict[3] + diag.hit_percentage_dict[4]
    
    return diag.peak_hflux, cap_hits