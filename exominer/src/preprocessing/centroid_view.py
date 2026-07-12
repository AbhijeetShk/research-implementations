import numpy as np


def compute_centroid_shift(
    centroid_x,
    centroid_y
):
    """
    Compute centroid displacement
    relative to the median position.
    """

    median_x = np.median(
        centroid_x
    )

    median_y = np.median(
        centroid_y
    )

    shift = np.sqrt(
        (centroid_x - median_x) ** 2
        +
        (centroid_y - median_y) ** 2
    )

    return shift


def generate_full_centroid_view(
    phase,
    centroid_shift,
    num_bins=50
):
    bin_edges = np.linspace(
        -0.5,
        0.5,
        num_bins + 1
    )

    binned_shift = np.zeros(
        num_bins
    )

    for i in range(num_bins):

        mask = (
            (phase >= bin_edges[i])
            &
            (phase < bin_edges[i + 1])
        )

        if np.any(mask):

            binned_shift[i] = np.mean(
                centroid_shift[mask]
            )

        else:

            binned_shift[i] = 0.0

    return binned_shift


def generate_local_centroid_view(
    phase,
    centroid_shift,
    phase_window=0.1,
    num_bins=50
):
    mask = (
        np.abs(phase)
        <= phase_window
    )

    local_phase = phase[mask]

    local_shift = (
        centroid_shift[mask]
    )

    bin_edges = np.linspace(
        -phase_window,
        phase_window,
        num_bins + 1
    )

    binned_shift = np.zeros(
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

            binned_shift[i] = np.mean(
                local_shift[bin_mask]
            )

        else:

            binned_shift[i] = 0.0

    return binned_shift