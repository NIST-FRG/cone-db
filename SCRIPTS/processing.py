from math import sqrt, exp


def calc_P_s(T_a):
    print(T_a)
    return 610.94 * exp((17.625 * T_a) / (T_a + 243.04))


def calculate_HRR_O2_only(
    X_O2, initial_X_O2, initial_X_CO2, delta_P, T_e, c, e, area, rh, T_a, P_a
):
    # for an ideal gas, volume percentage = mole fraction
    # assume for now



    M_O2 = 32  # 32 kg / kmol
    M_a = 28.96  # 28.96kg / kmol

    initial_X_H2O = (rh / 100) * (calc_P_s(T_a) / P_a)
    print(calc_P_s(T_a))
    print(initial_X_H2O)

    # Janssens 1991

    hrr = (
        e
        * 10**3
        * 1.1
        * c
        * sqrt(delta_P / T_e)
        * ((initial_X_O2 - X_O2) / (1.105 - (1.5 * X_O2)))
    )

    # hrr = (
    #     e
    #     * 10**3
    #     * ((initial_X_O2 - X_O2) / (1 - X_O2))
    #     * c
    #     * sqrt(delta_P / T_e)
    #     * (M_O2 / M_a)
    #     * (1 - initial_X_H2O - initial_X_CO2)
    # )

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
    e_co = 17.6  # mj/kg
    alpha = 1.105
    M_O2 = 32  # 32 kg / kmol i.e. 32g/mol
    M_a = 28.96  # 28.96kg / kmol

    # magnus formula (saturation water vapor pressure)
    P_s = 601.94 * exp((17.625 * T_a) / (T_a + 243.04))
    initial_X_H2O = (rel_humidity / 100) * (P_s / P_a)

    # oxygen depletion factor
    odf = (initial_X_O2 * (1 - X_CO2 - X_CO) - X_O2 * (1 - initial_X_CO2)) / (
        initial_X_O2 * (1 - X_CO2 - X_CO - X_O2)
    )

    duct_mass_flow_rate = c * sqrt(delta_P / T_e)

    # hrr = (
    #     1.10
    #     * e
    #     * initial_X_O2
    #     * duct_mass_flow_rate
    #     * ((odf - 0.172 * (1 - odf) * (X_CO / X_O2)) / (1 - odf + 1.105 * odf))
    # )

    hrr = (
        e * odf
        - ((e_co - e) * ((1 - odf) / 2) * (X_CO / X_O2))
        * (duct_mass_flow_rate / (1 + odf * (alpha - 1)))
        * (M_O2 / M_a)
        * (1 - initial_X_H2O)
        * initial_X_O2
    )

    # hrr per unit area (kW/m^2)
    return hrr / area
