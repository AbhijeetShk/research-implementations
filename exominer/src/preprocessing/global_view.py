import numpy as np


def generate_global_view(
    phase,
    flux,
    num_bins=200
):
    bin_edges = np.linspace(
        -0.5,
        0.5,
        num_bins + 1
    )

    binned_flux = np.zeros(
        num_bins
    )

    for i in range(num_bins):

        mask = (
            (phase >= bin_edges[i])
            &
            (phase < bin_edges[i + 1])
        )

        if np.any(mask):

            binned_flux[i] = np.mean(
                flux[mask]
            )

        else:

            binned_flux[i] = np.nan

    return binned_flux