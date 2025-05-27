def acceleration_from_velocity(v, vmax=250, k=0.085):
    """
    Returns the acceleration [kph/s] based on the instantaneous speed [kph].

    Parameters:
    - v: instantaneous speed [kph]
    - vmax: maximum speed [kph] (default 250)
    - k: acceleration rate constant (default 0.1)

    Returns:
    - acceleration [kph/s]
    """

    if v >= vmax:
        return 0.0
    return k * (vmax - v)
