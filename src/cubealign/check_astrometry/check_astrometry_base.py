#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#


from pathlib import Path
from reproject import reproject_interp
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

import logging
import matplotlib.pyplot as plt
import numpy as np

from .reproject import reproject_astrometric_image
from .plot_astrometry_images import plot_astrometry_images
from ..calculate_offset.align_image import prepare_images_for_offset, measure_offsets, pearson_images, align_image
from ..calculate_offset.plot_alignment_contours import plot_offset_comparison


logger = logging.getLogger(__name__)


def evaluate_alignment_methods(reference, moving, method="both", subtract_background=False):
    """Measure offsets and evaluate Pearson correlation for each Numina method."""

    ref, mov, common = prepare_images_for_offset(reference, moving)
    offsets = measure_offsets(ref, mov, method=method, subtract_background=subtract_background)

    r_original, n_original = pearson_images(reference, moving)
    logger.info("Original | Pearson = %.5f | Valid pixels = %d", r_original, n_original)

    results = {}

    for name, offset in offsets.items():
        aligned = align_image(moving, offset["dx"], offset["dy"])
        r, n_pixels = pearson_images(reference, aligned)

        results[name] = {
            "dx": offset["dx"], "dy": offset["dy"],
            "pearson": r, "n_pixels": n_pixels,
            "aligned_image": aligned
        }

        logger.info("%s | dx = %+.4f | dy = %+.4f | Pearson = %.5f | N = %d", name.upper(), offset["dx"], offset["dy"], r, n_pixels)

    return {"original_pearson": r_original, "original_pixels": n_original, "methods": results}


def refine_reference_astrometry(config, reference_image, reference_wcs, method, choose_method):
    """Measure the external astrometric offset of the reference cube."""

    astrometry = config["astrometry"]
    ref_id = config["ref_cube"]
    label = astrometry.get("label", "External image")

    logger.info("Starting external astrometric refinement.")

    cutout_config = astrometry.get("cutout", {})

    external_image, footprint, cutout_data, cutout_wcs = reproject_astrometric_image(
        astrometry["image"], astrometry["image_ext"], reference_wcs, reference_image.shape,
        margin=float(cutout_config.get("margin", 1.5)),
        padding_arcsec=float(cutout_config.get("padding_arcsec", 5.0))
    )

    # Diagnostic comparison before alignment
    plot_astrometry_images(reference_image, external_image, ref_id, label=label)

    # Require sufficient finite common coverage
    common = np.isfinite(reference_image) & np.isfinite(external_image) & (footprint > 0)
    common_fraction = np.count_nonzero(common) / reference_image.size

    logger.info("Common astrometric coverage: %.2f%%", 100 * common_fraction)

    if np.count_nonzero(common) < 10 or common_fraction < 0.1:
        raise ValueError("Insufficient common coverage for astrometric refinement.")

    # External image is the moving image
    results = evaluate_alignment_methods(
        reference_image, external_image, method=method, subtract_background=True
    )

    interactive = method == "both" and logger.isEnabledFor(logging.DEBUG)
    fig = None

    try:
        if logger.isEnabledFor(logging.INFO):
            fig = plot_offset_comparison(
                reference_image, external_image, results,
                label, ref_id, show=interactive
            )

        selected_method = choose_method(results, interactive=interactive) if method == "both" else method

    finally:
        if fig is not None:
            plt.close(fig)

    selected = results["methods"][selected_method]

    logger.info("Astrometry | Method: %s | dx = %+.4f | dy = %+.4f | Pearson = %.5f",
                selected_method.upper(), selected["dx"], selected["dy"], selected["pearson"])

    return {
        "method": selected_method,
        "dx": selected["dx"],
        "dy": selected["dy"],
        "pearson": selected["pearson"],
        "n_pixels": selected["n_pixels"],
        "aligned_image": selected["aligned_image"],
        "external_image": external_image,
        "footprint": footprint,
        "results": results
    }

