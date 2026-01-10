import numpy as np
from island_divertor_opt.postprocessing.heat_flux_diagnostic import HeatFluxDiagnostic

def get_cost_vars(path):
    diag = HeatFluxDiagnostic(path, verbose=False)
    diag.calculate_peak_hflux()
    qpeak = diag.peak_hflux
    diag.calculate_percentage()
    cap_hits = diag.hit_percentage_dict[3] + diag.hit_percentage_dict[4]
    
    return qpeak, cap_hits

def calculate_cost(qpeak, cap_hits, Q_COST=1, CAP_COST=1):
    return qpeak * Q_COST + cap_hits * CAP_COST


