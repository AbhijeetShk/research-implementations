import numpy as np


def phase_fold(
    time,
    flux,
    period
):
    phase = np.mod(
        time,
        period
    )

    sort_idx = np.argsort(
        phase
    )

    phase = phase[sort_idx]

    flux = flux[sort_idx]


    return phase, flux