#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from astropy.io import fits
import logging
import numpy as np

from astropy.table import Table


logger = logging.getLogger(__name__)

def read_line_table(filename, table_ext=1, cube_id=None):
    """Read a fitted emission-line FITS table."""

    table = Table.read(filename, format="fits", hdu=table_ext)

    if len(table) == 0:
        raise ValueError(f"{cube_id}: FITS table contains no rows.")

    logger.info("Table loaded: %s (%d rows)", cube_id, len(table))
    logger.debug("%s columns: %s", cube_id, table.colnames)

    return table



def inspect_table_coordinates(table, shape, col_names, cube_id, fits_coordinates=True):
    """Inspect and validate table coordinates against the cube spatial dimensions."""

    ny, nx = shape
    x = np.asarray(table[col_names["x_coord"]], dtype=float)
    y = np.asarray(table[col_names["y_coord"]], dtype=float)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{cube_id}: table contains non-finite coordinates.")

    x_min, x_max = (1, nx) if fits_coordinates else (0, nx - 1)
    y_min, y_max = (1, ny) if fits_coordinates else (0, ny - 1)

    logger.debug("%s | X range: %.2f - %.2f | Expected: %d - %d", cube_id, np.min(x), np.max(x), x_min, x_max)
    logger.debug("%s | Y range: %.2f - %.2f | Expected: %d - %d", cube_id, np.min(y), np.max(y), y_min, y_max)

    if np.any((x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)):
        raise ValueError(f"{cube_id}: table coordinates fall outside the cube spatial dimensions.")

    if not np.allclose(x, np.rint(x)) or not np.allclose(y, np.rint(y)):
        raise ValueError(f"{cube_id}: non-integer pixel coordinates detected.")

    logger.debug("%s: coordinate validation completed.", cube_id)



def inspect_table_header(filename, cube_id):
    """Inspect FITS table headers and available WCS keywords."""

    if not logger.isEnabledFor(logging.DEBUG):
        return

    keys = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2"]

    with fits.open(filename, memmap=True) as hdul:
        logger.debug("Inspecting table FITS file: %s", filename)

        for i, hdu in enumerate(hdul):
            logger.debug("%s | HDU %d: %s (%s)", cube_id, i, hdu.name, type(hdu).__name__)

            for key in keys:
                if key in hdu.header:
                    logger.debug("    %-8s = %s", key, hdu.header[key])

            table_wcs_keys = [key for key in hdu.header if key.startswith(("TCTYP", "TCRVL", "TCRPX", "TCDLT", "TCUNI", "TP", "TC"))]

            if table_wcs_keys:
                logger.debug("    Table-coordinate WCS keywords: %s", table_wcs_keys)



def table_to_image(table, shape, col_names, cube_id=None, fits_coordinates=True):
    """Build a 2D emission-line flux map from a FITS table."""

    ny, nx = shape
    image = np.full((ny, nx), np.nan, dtype=float)

    x = np.asarray(table[col_names["x_coord"]], dtype=float)
    y = np.asarray(table[col_names["y_coord"]], dtype=float)
    values = np.asarray(table[col_names["line_flux"]], dtype=float)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{cube_id}: non-finite pixel coordinates.")

    if not np.allclose(x, np.rint(x)) or not np.allclose(y, np.rint(y)):
        raise ValueError(f"{cube_id}: non-integer pixel coordinates.")

    ix = np.rint(x).astype(int) - int(fits_coordinates)
    iy = np.rint(y).astype(int) - int(fits_coordinates)

    if np.any((ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)):
        raise ValueError(f"{cube_id}: table contains coordinates outside the cube.")

    valid = np.isfinite(values)
    pixel_index = iy[valid] * nx + ix[valid]

    if len(np.unique(pixel_index)) != len(pixel_index):
        raise ValueError(f"{cube_id}: duplicate valid flux measurements for the same pixel.")

    image[iy[valid], ix[valid]] = values[valid]

    logger.info("%s: line map generated (%d finite pixels).", cube_id, np.count_nonzero(np.isfinite(image)))

    if not np.any(np.isfinite(image)):
        raise ValueError(f"{cube_id}: reconstructed line map has no finite pixels.")

    return image
