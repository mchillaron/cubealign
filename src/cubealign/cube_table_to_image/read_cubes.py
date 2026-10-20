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
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from pathlib import Path

logger = logging.getLogger(__name__)


def read_cube(filename, wcs_ext=0):
    """Read a FITS datacube and its WCS from the specified extension."""

    with fits.open(filename, memmap=True) as hdul:
        hdu = hdul[wcs_ext]

        if hdu.data is None or hdu.data.ndim != 3:
            raise ValueError(f"Expected a 3D datacube in {filename}, extension {wcs_ext}.")

        cube = hdu.data.copy()
        header = hdu.header.copy()

    wcs = WCS(header)

    if wcs.pixel_n_dim != 3 or not wcs.has_celestial or not wcs.has_spectral:
        raise ValueError(f"Invalid 3D celestial/spectral WCS in {filename}.")

    logger.debug("Cube: %s | Extension: %s | Shape: %s", filename, wcs_ext, cube.shape)

    return cube, header, wcs




def wavelength_axis(wcs, nlam):
    """Return the spectral wavelength axis in Angstrom."""

    try:
        spectral_wcs = wcs.spectral
        spectral_type = spectral_wcs.world_axis_physical_types[0]

        if spectral_type is None or not spectral_type.startswith("em.wl"):
            raise ValueError(f"Expected a wavelength axis, found: {spectral_type}")

        spectral_unit = u.Unit(spectral_wcs.world_axis_units[0])
        pixels = np.arange(nlam, dtype=float)
        wavelength = np.asarray(spectral_wcs.pixel_to_world_values(pixels), dtype=float)
        lam = (wavelength * spectral_unit).to_value(u.AA)

    except (ValueError, TypeError, u.UnitConversionError) as exc:
        raise ValueError(f"Cannot reconstruct wavelength axis: {exc}") from exc

    if not np.all(np.isfinite(lam)) or np.any(np.diff(lam) == 0):
        raise ValueError("Invalid or repeated wavelengths in the spectral WCS.")

    logger.debug("Spectral type: %s", spectral_type)
    logger.debug("Wavelength coverage: %.2f - %.2f Angstrom", np.min(lam), np.max(lam))

    return lam




def pixel_scales_arcsec(wcs, cube_id=None):
    """Return spatial pixel scales in arcsec/pixel."""

    celestial_wcs = wcs.celestial
    scales = proj_plane_pixel_scales(celestial_wcs)
    units = celestial_wcs.wcs.cunit

    scales = np.array([(scale * u.Unit(unit)).to_value(u.arcsec) for scale, unit in zip(scales, units)])

    logger.debug("%s pixel scale: X = %.5f, Y = %.5f arcsec/pix", cube_id or "Cube", scales[0], scales[1])

    return scales


def log_wcs_info(header, cube_id):
    """Log selected FITS WCS keywords at DEBUG level."""

    if not logger.isEnabledFor(logging.DEBUG):
        return

    keys = ["NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CTYPE1", "CTYPE2"]

    logger.debug("WCS information for %s:", cube_id)

    for key in keys:
        logger.debug("  %-8s: %s", key, header.get(key, "Not defined"))




