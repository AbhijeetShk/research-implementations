import numpy as np


def extract_scalar_features(
    flux,
    period,
    duration
):
    transit_depth = (
        np.max(flux)
        - np.min(flux)
    )

    flux_mean = np.mean(
        flux
    )

    flux_std = np.std(
        flux
    )

    return np.array(
        [
            period,
            duration,
            transit_depth,
            flux_mean,
            flux_std
        ],
        dtype=np.float32
    )