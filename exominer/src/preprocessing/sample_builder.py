from src.preprocessing.transit_centered_fold import (
    transit_centered_fold
)

from src.preprocessing.global_view import (
    generate_global_view
)

from src.preprocessing.local_view import (
    generate_local_view
)

from src.preprocessing.centroid_view import (
    compute_centroid_shift,
    generate_full_centroid_view,
    generate_local_centroid_view
)

from src.preprocessing.scalar_features import (
    extract_scalar_features
)


def build_exominer_sample(
    time,
    flux,
    centroid_x,
    centroid_y,
    period,
    epoch,
    duration
):


    phase, folded_flux = (
        transit_centered_fold(
            time,
            flux,
            period,
            epoch
        )
    )

    full_flux = generate_global_view(
        phase,
        folded_flux
    )

    transit_flux = generate_local_view(
        phase,
        folded_flux
    )



    centroid_shift = (
        compute_centroid_shift(
            centroid_x,
            centroid_y
        )
    )

    phase_centroid, folded_shift = (
        transit_centered_fold(
            time,
            centroid_shift,
            period,
            epoch
        )
    )

    full_centroid = (
        generate_full_centroid_view(
            phase_centroid,
            folded_shift
        )
    )

    transit_centroid = (
        generate_local_centroid_view(
            phase_centroid,
            folded_shift
        )
    )



    scalar_features = (
        extract_scalar_features(
            folded_flux,
            period,
            duration
        )
    )

    return {
        "full_flux": full_flux,
        "transit_flux": transit_flux,
        "full_centroid": full_centroid,
        "transit_centroid": transit_centroid,
        "scalar_features": scalar_features
    }