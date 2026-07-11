import numpy as np


def generate_local_view(
    phase,
    flux,
    phase_window=0.1,
    num_bins=50
):
    mask = (
        np.abs(phase)
        <= phase_window
    )

    local_phase = phase[mask]
    local_flux = flux[mask]

    bin_edges = np.linspace(
        -phase_window,
        phase_window,
        num_bins + 1
    )

    binned_flux = np.zeros(
        num_bins
    )

    for i in range(num_bins):

        bin_mask = (
            (local_phase >= bin_edges[i])
            &
            (
                local_phase
                < bin_edges[i + 1]
            )
        )

        if np.any(bin_mask):

            binned_flux[i] = np.mean(
                local_flux[bin_mask]
            )

        else:

            binned_flux[i] = 1.0

    return binned_flux