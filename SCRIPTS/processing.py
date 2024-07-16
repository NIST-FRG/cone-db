from math import sqrt, exp


def calculate_HRR_O2_only(X_O2, X_O2_initial, delta_P, T_e, c, e, area):

    hrr = (
        e
        * 10**3
        * 1.1
        * c
        * sqrt(delta_P / T_e)
        * ((X_O2_initial - X_O2) / (1.105 - (1.5 * X_O2)))
    )

    # hrr per unit area (kW/m^2)
    return hrr / area


def calculate_MFR(c, delta_P, T_e):
    return c * sqrt(delta_P / T_e)

def calculate_k(I_0, )

def calculate_HRR(
    X_O2,
    X_CO2,
    X_CO,
    X_O2_initial,
    X_CO2_initial,
    delta_P,
    T_e,
    c,
    e,
    area,
):

    # oxygen depletion factor
    odf = (X_O2_initial * (1 - X_CO2 - X_CO) - X_O2 * (1 - X_CO2_initial)) / (
        X_O2_initial * (1 - X_CO2 - X_CO - X_O2)
    )

    duct_mass_flow_rate = calculate_MFR(c, delta_P, T_e)

    hrr = (
        1.10
        * e
        * 10**3
        * X_O2_initial
        * duct_mass_flow_rate
        * ((odf - 0.172 * (1 - odf) * (X_CO / X_O2)) / (1 - odf + 1.105 * odf))
    )

    # hrr per unit area (kW/m^2)
    return hrr / area
