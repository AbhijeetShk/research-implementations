import numpy as np


def transit_centered_fold(
    time,
    flux,
    period,
    epoch
):
    

    phase = (
        ((time - epoch + 0.5 * period) % period)
        / period
    ) - 0.5

    sort_idx = np.argsort(phase)

    phase = phase[sort_idx]
    folded_flux = flux[sort_idx]

    return phase, folded_flux