#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

import logging
import matplotlib.pyplot as plt

from ..initial_functions import choose_method
from .plot_alignment_contours import plot_offset_comparison, plot_final_alignment
from .align_image import evaluate_alignment_methods

logger = logging.getLogger(__name__)


def calculate_offset_base(config, images, wcs_dict, method,):

    # Prepare reference image and alignment results
    ref_id = config["ref_cube"]
    reference_image = images[ref_id]
    reference_wcs = wcs_dict[ref_id]

    logger.info("Reference cube: %s", ref_id)

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

        moving_image = images[cube_id]
        results = evaluate_alignment_methods(reference_image, moving_image, method=method)
        alignment_results[cube_id] = results

        # Display comparison and select method
        interactive = method == "both" and logger.isEnabledFor(logging.DEBUG)
        fig = None

        try:
            if logger.isEnabledFor(logging.INFO):
                fig = plot_offset_comparison(
                    reference_image, moving_image, results,
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

        aligned_images[cube_id] = selected["aligned_image"]

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
    
    # # Check completeness of relative alignment
    # expected_ids = {cube["id"] for cube in config["cubes"]}
    # processed_ids = set(aligned_images)
    # if processed_ids != expected_ids:
    #     raise RuntimeError(f"Alignment incomplete. Missing cubes: {sorted(expected_ids - processed_ids)}")

    # # Verify WCS compatibility before updating CRPIX
    # ref_shape = reference_image.shape

    # for cube_info in config["cubes"]:
    #     cube_id = cube_info["id"]

    #     if images[cube_id].shape != ref_shape:
    #         raise ValueError(f"{cube_id}: spatial shape differs from reference. Reprojection required.")

    #     if cube_id != ref_id:
    #         validate_wcs_for_crpix_update(reference_wcs, wcs_dict[cube_id], ref_shape)

    # final alignment visualization
    plot_final_alignment(images, aligned_images, config["cubes"], ref_id)

    return reference_image, reference_wcs, alignment_results, offsets, aligned_images
