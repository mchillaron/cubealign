#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from scipy.ndimage import shift

import logging
import matplotlib.pyplot as plt
import numpy as np

from ..initial_functions import choose_method
from .plot_alignment_contours import plot_offset_comparison, plot_final_alignment
from .align_image import evaluate_alignment_methods
from .normalize import normalize_alignment_image
from .smoothing import smooth_alignment_image

logger = logging.getLogger(__name__)


def preprocess_alignment_images(images, normalize=False, smooth_kernel=None):
    """Prepare image copies for offset measurement without modifying original data."""

    processed = {}

    for cube_id, image in images.items():
        data = np.asarray(image, dtype=float).copy()

        if smooth_kernel is not None:
            data = smooth_alignment_image(data, smooth_kernel, cube_id)

        if normalize:
            data = normalize_alignment_image(data, cube_id)

        processed[cube_id] = data

    return processed



def align_image(image, dx, dy, order=3):
    """Shift a 2D image using Numina offsets, preserving invalid pixels."""

    if image.ndim != 2:
        raise ValueError("align_image requires a 2D image.")

    if not np.all(np.isfinite([dx, dy])):
        raise ValueError("Alignment offsets must be finite.")

    finite = np.isfinite(image)
    data = np.where(finite, image, 0.0)
    weights = finite.astype(float)

    shifted_data = shift(data, shift=(-dy, -dx), order=order, mode="constant", cval=0.0, prefilter=order > 1)
    shifted_weights = shift(weights, shift=(-dy, -dx), order=1, mode="constant", cval=0.0, prefilter=False)

    aligned = np.full(image.shape, np.nan, dtype=float)
    valid = shifted_weights > 0.999
    aligned[valid] = shifted_data[valid] / shifted_weights[valid]

    return aligned



def calculate_offset_base(config, images, wcs_dict, method, normalize=False, smooth_kernel=None,
                          tukey=None, rescale_to_01=True):

    # Prepare reference image and alignment results
    ref_id = config["ref_cube"]
    reference_image = images[ref_id]
    reference_wcs = wcs_dict[ref_id]

    logger.info("Reference cube: %s", ref_id)

    processed_images = preprocess_alignment_images(images, normalize=normalize, smooth_kernel=smooth_kernel)
    processed_reference = processed_images[ref_id]

    offsets = {}
    aligned_images = {ref_id: reference_image.copy()}
    alignment_results = {}

    # Measure offsets for each moving cube
    for cube_info in config["cubes"]:
        cube_id = cube_info["id"]

        if cube_id == ref_id:
            continue

        logger.info("=" * 55)
        logger.info("Aligning %s relative to %s", cube_id, ref_id)
        logger.info("=" * 55)

        moving_image = processed_images[cube_id]
        
        results = evaluate_alignment_methods(
            processed_reference, moving_image, method=method,
            tukey=tukey, rescale_to_01=rescale_to_01,
            cube_id=cube_id, ref_id=ref_id,
            plot_window=logger.isEnabledFor(logging.INFO),
            show_window=logger.isEnabledFor(logging.DEBUG),
        )
        alignment_results[cube_id] = results

        # Display comparison and select method
        interactive = method == "both" and logger.isEnabledFor(logging.DEBUG)
        fig = None

        try:
            if logger.isEnabledFor(logging.INFO):
                fig = plot_offset_comparison(
                    processed_reference, moving_image, results,  #reference_image
                    cube_id, ref_id, show=interactive
                )

            if method == "both":
                selected_method = choose_method(results, interactive=interactive)
            else:
                selected_method = method

        finally:
            if fig is not None:
                plt.close(fig)

        selected = results["methods"][selected_method]

        offsets[cube_id] = {
            "method": selected_method,
            "dx": selected["dx"],
            "dy": selected["dy"],
            "pearson": selected["pearson"],
            "n_pixels": selected["n_pixels"],
        }

        #aligned_images[cube_id] = selected["aligned_image"]
        aligned_images[cube_id] = align_image(images[cube_id], selected["dx"], selected["dy"])

        logger.info(
            "%s | Selected %s | dx = %+.4f | dy = %+.4f | Pearson = %.5f",
            cube_id, selected_method.upper(),
            selected["dx"], selected["dy"], selected["pearson"]
        )

    # check all cubes have been processed 
    expected_ids = {cube["id"] for cube in config["cubes"]}
    processed_ids = set(aligned_images)

    if processed_ids != expected_ids:
        missing = expected_ids - processed_ids
        raise RuntimeError(f"Alignment incomplete. Missing cubes: {sorted(missing)}")

    logger.info("All %d cubes successfully processed.", len(expected_ids))
    
    # Final alignment using original images
    plot_final_alignment(images, aligned_images, config["cubes"], ref_id, cmap="plasma")

    # Additional plot using normalized images
    if normalize:
        logger.info("Generating normalized alignment comparison.")
        normalized_images = {cube_id: normalize_alignment_image(image, cube_id) for cube_id, image in images.items()}
        normalized_aligned = {ref_id: normalized_images[ref_id].copy()}
        for cube_id, solution in offsets.items():
            normalized_aligned[cube_id] = align_image(normalized_images[cube_id], solution["dx"], solution["dy"])

        plot_final_alignment(
            normalized_images, normalized_aligned, config["cubes"], ref_id,
            cmap="plasma", normalized=True, filename="final_alignment_normalized.png"
        )

    return reference_image, reference_wcs, alignment_results, offsets, aligned_images
