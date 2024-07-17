from math import sqrt, exp, log


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


def calculate_k(I_0, I, L):
    return log(I_0 / I) / L


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


def colorize(text, color):
    if color == "red":
        color = 196
    elif color == "green":
        color = 46
    elif color == "blue":
        color = 45
    elif color == "yellow":
        color = 226
    elif color == "purple":
        color = 99
    elif color == "cyan":
        color = 123
    elif color == "white":
        color = 15
    elif color == "black":
        color = 0
    else:
        color = 15

    return f"\x1b[38;5;{color}m{text}\x1b[0m"
