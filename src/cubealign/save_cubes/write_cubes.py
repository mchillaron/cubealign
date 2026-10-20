#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path

import logging
import numpy as np

logger = logging.getLogger(__name__)


def write_aligned_cubes(config, corrections, offsets, astrometry_result, history_line_command, output_dir=None):
    """Write new FITS cubes with corrected CRPIX1/2, preserving original files."""

    output_dir = Path(output_dir or Path.cwd()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_id = config["ref_cube"]
    written = {}

    # Verify that every output can be created before writing any file
    targets = {}

    for cube_info in config["cubes"]:
        cube_id = cube_info["id"]
        source = Path(cube_info["path"])
        target = output_dir / f"{source.stem}_aligned.fits"

        targets[cube_id] = target

    for cube_info in config["cubes"]:
        cube_id = cube_info["id"]
        source = Path(cube_info["path"])
        target = targets[cube_id]
        correction = corrections[cube_id]
        ext = cube_info["wcs_ext"]

        with fits.open(source, memmap=False) as hdul:
            header = hdul[ext].header

            if "CRPIX1" not in header or "CRPIX2" not in header:
                raise ValueError(f"{cube_id}: CRPIX1/2 not found in the selected WCS extension.")

            old_crpix1 = float(header["CRPIX1"])
            old_crpix2 = float(header["CRPIX2"])

            header["CRPIX1"] = old_crpix1 + correction["dcrpix1"]
            header["CRPIX2"] = old_crpix2 + correction["dcrpix2"]

            header.add_history("Cube spatial alignment and astrometric correction")
            header.add_history(f"Reference cube: {ref_id}")
            header.add_history(f"Relative offset: dx={correction['dx_relative']:+.6f}, dy={correction['dy_relative']:+.6f} pix")

            if cube_id != ref_id:
                header.add_history(f"Relative alignment method: {offsets[cube_id]['method']}")

            if astrometry_result is not None:
                header.add_history(f"External astrometry method: {astrometry_result['method']}")
                header.add_history(f"External image offset: dx={correction['dx_astrometry']:+.6f}, dy={correction['dy_astrometry']:+.6f} pix")
                header.add_history(f"External image: {config['astrometry']['image']}")

            header.add_history(f"CRPIX1: {old_crpix1:.8f} -> {header['CRPIX1']:.8f}")
            header.add_history(f"CRPIX2: {old_crpix2:.8f} -> {header['CRPIX2']:.8f}")
            header.add_history(f"Command: {history_line_command}")

            hdul.writeto(target, overwrite=True, checksum=True)

        written[cube_id] = target
        logger.info("Aligned cube saved: %s", target)

    return written



def write_aligned_tables(config, corrections, wcs_dict, output_dir=None):
    """Write corrected FITS tables only when their image-style WCS can be verified."""

    if config["mode"] != "tables":
        return {}

    output_dir = Path(output_dir or Path.cwd()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    for table_info in config["tables"]:
        cube_id = table_info["cube_id"]
        source = Path(table_info["path"])
        target = output_dir / f"{source.stem}_{cube_id}_aligned.fits"

        with fits.open(source, memmap=False) as hdul:
            updated = []

            for i, hdu in enumerate(hdul):
                header = hdu.header

                required = ("CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2")

                if not all(key in header for key in required):
                    continue

                try:
                    table_wcs = WCS(header).celestial

                    if not table_wcs.has_celestial or table_wcs.pixel_n_dim != 2:
                        continue

                    reference_wcs = wcs_dict[cube_id].celestial
                    shape = next(cube for cube in config["cubes"] if cube["id"] == cube_id)

                    # Check WCS agreement at several image positions
                    with fits.open(shape["path"], memmap=True) as cube_hdul:
                        ny, nx = cube_hdul[shape["wcs_ext"]].data.shape[-2:]

                    xx = np.array([0, nx / 2, nx - 1], dtype=float)
                    yy = np.array([0, ny / 2, ny - 1], dtype=float)
                    sky = table_wcs.pixel_to_world(xx, yy)
                    xr, yr = reference_wcs.world_to_pixel(sky)

                    if np.max(np.hypot(xr - xx, yr - yy)) > 0.05:
                        logger.warning("%s HDU %d: table WCS differs from cube WCS. Skipping.", cube_id, i)
                        continue

                except (ValueError, TypeError, KeyError) as exc:
                    logger.warning("%s HDU %d: cannot validate table WCS: %s", cube_id, i, exc)
                    continue

                header["CRPIX1"] += corrections[cube_id]["dcrpix1"]
                header["CRPIX2"] += corrections[cube_id]["dcrpix2"]
                header.add_history("Celestial WCS updated using associated cube alignment.")
                updated.append(i)

            if not updated:
                logger.warning("%s: no compatible image-style WCS found. Original table left unchanged.", cube_id)
                continue

            hdul.writeto(target, overwrite=True, checksum=True)

        written[cube_id] = target
        logger.info("Aligned FITS table saved: %s (HDUs: %s)", target, updated)

    return written
