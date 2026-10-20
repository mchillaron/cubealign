#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#


from numina.tools.measure_xy_offset_2dimages import measure_xy_offset_2dimages
from scipy.ndimage import shift
from scipy.stats import pearsonr

import logging
import numpy as np


logger = logging.getLogger(__name__)


def prepare_images_for_offset(reference, moving):
    """Prepare two equally shaped images for Numina cross-correlation."""

    if reference.ndim != 2 or moving.ndim != 2:
        raise ValueError("Offset measurement requires two 2D images.")

    if reference.shape != moving.shape:
        raise ValueError(f"Images have different shapes: {reference.shape} and {moving.shape}. Reprojection is required.")

    common = np.isfinite(reference) & np.isfinite(moving)

    if np.count_nonzero(common) < 3:
        raise ValueError("Insufficient common finite pixels for alignment.")

    ref = np.zeros(reference.shape, dtype=float)
    mov = np.zeros(moving.shape, dtype=float)

    ref[common] = reference[common]
    mov[common] = moving[common]

    return ref, mov, common


def measure_offsets(reference, moving, method="both", subtract_background=False, rescale_to_01=True):
    """Measure image offsets using one or both Numina methods."""

    methods = [1, 2] if method == "both" else [int(method[-1])]
    results = {}

    for m in methods:
        dx, dy = measure_xy_offset_2dimages(
            image1=reference, image2=moving,
            subtract_background=subtract_background,
            rescale_to_01=rescale_to_01,
            method=m, plots=False
        )

        if not np.all(np.isfinite([dx, dy])):
            raise ValueError(f"Numina method {m} returned non-finite offsets.")

        results[f"m{m}"] = {"dx": float(dx), "dy": float(dy)}
        logger.info("Method M%d | dx = %+.4f pix | dy = %+.4f pix", m, dx, dy)

    return results


def align_image(image, dx, dy, order=3):
    """Shift an image using Numina offsets while preserving invalid regions."""

    finite = np.isfinite(image)
    data = np.where(finite, image, 0.0)
    weights = finite.astype(float)

    shifted_data = shift(data, shift=(-dy, -dx), order=order, mode="constant", cval=0.0, prefilter=order > 1)
    shifted_weights = shift(weights, shift=(-dy, -dx), order=1, mode="constant", cval=0.0, prefilter=False)

    aligned = np.full(image.shape, np.nan, dtype=float)
    valid = shifted_weights > 0.999

    aligned[valid] = shifted_data[valid] / shifted_weights[valid]

    return aligned


def pearson_images(image1, image2, min_pixels=10):
    """Return Pearson correlation and number of common valid pixels."""

    if image1.shape != image2.shape:
        raise ValueError("Pearson correlation requires images with identical shapes.")

    valid = np.isfinite(image1) & np.isfinite(image2)
    n_pixels = np.count_nonzero(valid)

    if n_pixels < min_pixels:
        return np.nan, n_pixels

    x, y = image1[valid], image2[valid]

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, n_pixels

    r, _ = pearsonr(x, y)

    return float(r), n_pixels


def evaluate_alignment_methods(reference, moving, method="both"):
    """Measure offsets and evaluate the alignment quality for each method."""

    ref, mov, common = prepare_images_for_offset(reference, moving)
    offsets = measure_offsets(ref, mov, method=method)

    r_original, n_original = pearson_images(reference, moving)
    logger.info("Original | Pearson = %.5f | Valid pixels = %d", r_original, n_original)

    results = {}

    for name, offset in offsets.items():
        aligned = align_image(moving, offset["dx"], offset["dy"])
        r, n_pixels = pearson_images(reference, aligned)

        results[name] = {
            "dx": offset["dx"],
            "dy": offset["dy"],
            "pearson": r,
            "n_pixels": n_pixels,
            "aligned_image": aligned,
        }

        logger.info("%s | dx = %+.4f | dy = %+.4f | Pearson = %.5f | N = %d", name.upper(), offset["dx"], offset["dy"], r, n_pixels)

    return {"original_pearson": r_original, "original_pixels": n_original, "methods": results}