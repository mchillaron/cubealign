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

import logging
import numpy as np

logger = logging.getLogger(__name__)

from .cutout_external_image import cutout_external_image

def reproject_astrometric_image(filename, image_ext, reference_wcs, reference_shape, margin=1.5, padding_arcsec=5.0):
    """Cut out and reproject an external FITS image onto the reference cube grid."""

    with fits.open(filename, memmap=True) as hdul:
        hdu = hdul[image_ext]

        if hdu.data is None or hdu.data.ndim != 2:
            raise ValueError(f"Expected a 2D astrometric FITS image: {filename}")

        data = np.asarray(hdu.data, dtype=float).copy()
        external_wcs = WCS(hdu.header).celestial

    if not external_wcs.has_celestial:
        raise ValueError("External image has no valid celestial WCS.")

    logger.info("Reading external astrometric image: %s", filename)

    # 1. Extract external-image cutout using the reference cube WCS
    cutout_data, cutout_wcs = cutout_external_image(
        data, external_wcs, reference_wcs, reference_shape,
        margin=margin, padding_arcsec=padding_arcsec
    )

    if not np.any(np.isfinite(cutout_data)):
        raise ValueError("External cutout contains no finite pixels.")

    # 2. Reproject cutout onto the reference cube spatial grid
    reprojected, footprint = reproject_interp(
        (cutout_data, cutout_wcs), reference_wcs.celestial,
        shape_out=reference_shape, order="bilinear"
    )

    reprojected[~np.isfinite(footprint) | (footprint <= 0)] = np.nan

    valid = np.isfinite(reprojected)
    coverage = np.count_nonzero(valid) / reprojected.size

    logger.info("External cutout shape: %s", cutout_data.shape)
    logger.info("Reprojected shape: %s", reprojected.shape)
    logger.info("External coverage: %.2f%%", 100 * coverage)

    if np.count_nonzero(valid) < 10:
        raise ValueError("Insufficient external-image coverage after reprojection.")

    return reprojected, footprint, cutout_data, cutout_wcs


def reproject_astrometric_image_old(filename, image_ext, reference_wcs, reference_shape):
    """Reproject an external FITS image onto the reference cube's celestial grid."""

    with fits.open(filename, memmap=True) as hdul:
        hdu = hdul[image_ext]

        if hdu.data is None or hdu.data.ndim != 2:
            raise ValueError(f"Astrometric reference must be a 2D FITS image: {filename}")

        data = np.asarray(hdu.data, dtype=float).copy()
        external_wcs = WCS(hdu.header).celestial

    if not external_wcs.has_celestial:
        raise ValueError("External image has no valid celestial WCS.")

    logger.info("External image: %s | Shape: %s", filename, data.shape)

    reprojected, footprint = reproject_interp(
        (data, external_wcs), reference_wcs.celestial,
        shape_out=reference_shape, order="bilinear"
    )

    reprojected[~np.isfinite(footprint) | (footprint <= 0)] = np.nan
    valid = np.isfinite(reprojected)

    logger.info("Reprojected image shape: %s", reprojected.shape)
    logger.info("Valid external coverage: %.2f%%", 100 * np.count_nonzero(valid) / valid.size)

    if np.count_nonzero(valid) < 10:
        raise ValueError("Insufficient external-image coverage of the reference cube.")

    return reprojected, footprint, data, external_wcs
