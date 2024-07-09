from math import sqrt, exp

def calculate_HRR_O2_only(
    X_O2, initial_X_O2, initial_X_CO2, delta_P, T_e, c, e, area, rh, T_a, P_a
):
    # for an ideal gas, volume percentage = mole fraction
    # assume for now

    # Janssens 1991

    hrr = (
        e
        * 10**3
        * 1.1
        * c
        * sqrt(delta_P / T_e)
        * ((initial_X_O2 - X_O2) / (1.105 - (1.5 * X_O2)))
    )

    # hrr per unit area (kW/m^2)
    return hrr / area


def calculate_HRR(
    X_O2,
    X_CO2,
    X_CO,
    initial_X_O2,
    initial_X_CO2,
    delta_P,
    T_e,
    c,
    e,
    area,
    rel_humidity,
    T_a,
    P_a,
):
    # for an ideal gas, volume percentage = mole fraction
    # assume for now

    # e = 13.1 #mj/kg

    # oxygen depletion factor
    odf = (initial_X_O2 * (1 - X_CO2 - X_CO) - X_O2 * (1 - initial_X_CO2)) / (
        initial_X_O2 * (1 - X_CO2 - X_CO - X_O2)
    )

    duct_mass_flow_rate = c * sqrt(delta_P / T_e)

    hrr = (
        1.10
        * e
        * 10**3
        * initial_X_O2
        * duct_mass_flow_rate
        * ((odf - 0.172 * (1 - odf) * (X_CO / X_O2)) / (1 - odf + 1.105 * odf))
    )

    # hrr per unit area (kW/m^2)
    return hrr / area
