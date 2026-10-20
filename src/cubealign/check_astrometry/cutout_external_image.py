#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp

import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def cutout_external_image(data, external_wcs, reference_wcs, reference_shape, margin=1.5, padding_arcsec=5.0):
    """Extract an external-image cutout covering the reference cube footprint."""

    if margin < 1 or padding_arcsec < 0:
        raise ValueError("margin must be >= 1 and padding_arcsec must be >= 0.")

    ny, nx = reference_shape
    ref_wcs = reference_wcs.celestial
    ext_wcs = external_wcs.celestial

    # Reference cube geometric center in celestial coordinates
    center = ref_wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)

    # Sample the four edges to account for image orientation
    t = np.linspace(0, 1, 20)
    x_border = np.concatenate([t * (nx - 1), t * (nx - 1), np.zeros_like(t), np.full_like(t, nx - 1)])
    y_border = np.concatenate([np.zeros_like(t), np.full_like(t, ny - 1), t * (ny - 1), t * (ny - 1)])

    sky_border = ref_wcs.pixel_to_world(x_border, y_border)
    x_ext, y_ext = ext_wcs.world_to_pixel(sky_border)
    xc, yc = ext_wcs.world_to_pixel(center)

    if not np.all(np.isfinite(x_ext)) or not np.all(np.isfinite(y_ext)) or not np.all(np.isfinite([xc, yc])):
        raise ValueError("Cannot project the reference cube footprint onto the external image.")

    # Angular padding converted to external-image pixels
    scales = proj_plane_pixel_scales(ext_wcs)
    scale_x = (scales[0] * u.Unit(ext_wcs.wcs.cunit[0])).to_value(u.arcsec)
    scale_y = (scales[1] * u.Unit(ext_wcs.wcs.cunit[1])).to_value(u.arcsec)

    half_width = np.max(np.abs(x_ext - xc)) * margin + padding_arcsec / scale_x
    half_height = np.max(np.abs(y_ext - yc)) * margin + padding_arcsec / scale_y

    size = (max(2, int(np.ceil(2 * half_height))), max(2, int(np.ceil(2 * half_width))))

    cutout = Cutout2D(data, position=center, size=size, wcs=ext_wcs, mode="partial", fill_value=np.nan, copy=True)

    logger.info("Reference cube center: RA = %.7f deg, Dec = %.7f deg", center.icrs.ra.deg, center.icrs.dec.deg)
    logger.info("External image shape: %s | Cutout shape: %s", data.shape, cutout.data.shape)

    return cutout.data, cutout.wcs
