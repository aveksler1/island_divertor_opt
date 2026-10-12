import numpy as np
from island_divertor_opt.postprocessing.heat_flux_diagnostic import HeatFluxDiagnostic

def get_cost_vars(path):
    diag = HeatFluxDiagnostic(path, verbose=False)
    diag.calculate_peak_hflux()
    qpeak = diag.peak_hflux
    diag.calculate_percentage()
    cap_hits = diag.hit_percentage_dict[3] + diag.hit_percentage_dict[4]
    
    return qpeak, cap_hits

def get_cost_vars_resubmission(path):
    diag = HeatFluxDiagnostic(path, verbose=False)
    diag.calculate_peak_hflux()
    qpeak = diag.peak_hflux
    diag.calculate_percentage()
    cap_hits = diag.hit_percentage_dict[3] + diag.hit_percentage_dict[4]
    
    wall_loss = diag.hit_percentage_dict[0]

    return qpeak, cap_hits, wall_loss

def get_qpeak_wall(path):
    diag = HeatFluxDiagnostic(path, verbose=False)
    diag.calculate_hflux_wall()
    
    return diag.peak_wall_hflux

def calculate_cost(qpeak, cap_hits, Q_COST=1, CAP_COST=1):
    return qpeak * Q_COST + cap_hits * CAP_COST

def calculate_cost_power_fraction(qpeak, cap_hits, power_fraction):
    return qpeak + power_fraction

def calculate_power_fraction_captured_no_cap(path):
    diag = HeatFluxDiagnostic(path, verbose=False)
    diag.calculate_percentage()
    wall_pct = diag.hit_percentage_dict[0]
    cap_pct = diag.hit_percentage_dict[3] + diag.hit_percentage_dict[4]
    power_fraction = (100 - wall_pct - cap_pct) / 100
    print(f"power fraction = {power_fraction}")

    return power_fraction

def calculate_power_fraction_captured(path):
    """
    Returns the power fraction absorbed by the divertor versus the wall. 
    """
    diag = HeatFluxDiagnostic(path, verbose=False)
    diag.calculate_percentage()
    power_fraction = (100 - diag.hit_percentage_dict[0]) / 100
    print(f"power fraction = {power_fraction}")

    return power_fraction
