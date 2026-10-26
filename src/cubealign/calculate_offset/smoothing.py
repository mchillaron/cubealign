#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from scipy.ndimage import gaussian_filter

import logging
import numpy as np

logger = logging.getLogger(__name__)

def smooth_alignment_image(image, sigma, cube_id=None):
    """Apply NaN-aware Gaussian smoothing to a 2D alignment image."""

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive.")

    finite = np.isfinite(image)
    data = np.where(finite, image, 0.0)
    weights = finite.astype(float)

    smooth_data = gaussian_filter(data, sigma=sigma, mode="constant", cval=0.0)
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0)

    smoothed = np.full(image.shape, np.nan, dtype=float)
    valid = smooth_weights > 0.01

    smoothed[valid] = smooth_data[valid] / smooth_weights[valid]
    smoothed[~finite] = np.nan

    logger.debug("%s | Gaussian smoothing applied: sigma = %.3f pix", cube_id, sigma)

    return smoothed
