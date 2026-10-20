#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

import logging
import numpy as np

logger = logging.getLogger(__name__)

def validate_wcs_for_crpix_update(reference_wcs, moving_wcs, shape, tolerance_pix=0.05):
    """Require equivalent initial celestial WCS for direct CRPIX offset addition."""

    ref = reference_wcs.celestial
    mov = moving_wcs.celestial

    if reference_wcs.pixel_n_dim != 3 or moving_wcs.pixel_n_dim != 3:
        raise ValueError("Direct CRPIX correction currently requires 3D cube WCS.")

    if ref.sip is not None or mov.sip is not None:
        raise ValueError("SIP-distorted WCS requires a dedicated astrometric correction.")

    if list(ref.wcs.ctype) != list(mov.wcs.ctype):
        raise ValueError("Different celestial projections: direct CRPIX addition is unsafe.")

    ny, nx = shape
    xx = np.array([0, nx / 2, nx - 1, 0, nx - 1], dtype=float)
    yy = np.array([0, ny / 2, ny - 1, ny - 1, 0], dtype=float)

    sky = mov.pixel_to_world(xx, yy)
    xr, yr = ref.world_to_pixel(sky)
    residual = np.hypot(xr - xx, yr - yy)

    if not np.all(np.isfinite(residual)) or np.max(residual) > tolerance_pix:
        raise ValueError(
            f"Cube WCS differs from the reference by up to {np.max(residual):.4f} pixels. "
            "Reprojection or a general WCS transformation is required."
        )

    return True
