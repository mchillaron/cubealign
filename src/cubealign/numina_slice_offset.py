#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from astropy.io import fits
from pathlib import Path

import logging
import numpy as np
import subprocess
import shutil

logger = logging.getLogger(__name__)



def inspect_cube_slices(cube, cube_id):
    """Inspect non-finite values and spatial variability along the spectral axis."""

    if cube.ndim != 3:
        raise ValueError(f"{cube_id}: expected a 3D cube.")

    nlam, ny, nx = cube.shape
    finite = np.isfinite(cube)

    valid_pixels = np.sum(finite, axis=(1, 2))
    invalid_pixels = ny * nx - valid_pixels
    finite_fraction = valid_pixels / (ny * nx)

    with np.errstate(invalid="ignore"):
        spatial_std = np.nanstd(np.where(finite, cube, np.nan), axis=(1, 2))

    empty = np.where(valid_pixels == 0)[0]
    constant = np.where((valid_pixels > 0) & (spatial_std <= 1e-12))[0]
    incomplete = np.where(invalid_pixels > 0)[0]

    logger.info("%s | Spectral channels: %d", cube_id, nlam)
    logger.info("%s | Channels containing non-finite pixels: %d", cube_id, len(incomplete))
    logger.info("%s | Completely invalid channels: %d", cube_id, len(empty))
    logger.info("%s | Near-constant channels: %d", cube_id, len(constant))

    if len(empty):
        logger.warning("%s | First completely invalid channels: %s", cube_id, (empty[:10] + 1).tolist())

    if len(constant):
        logger.warning("%s | First near-constant channels: %s", cube_id, (constant[:10] + 1).tolist())

    return {"finite_fraction": finite_fraction, "spatial_std": spatial_std, "empty_channels": empty, "constant_channels": constant}



def prepare_cube_for_slice_diagnostic(cube, cube_id):
    """Prepare a diagnostic cube by replacing non-finite pixels channel by channel."""

    data = np.asarray(cube, dtype=float).copy()
    nlam = data.shape[0]
    n_replaced = 0

    for k in range(nlam):
        image = data[k]
        finite = np.isfinite(image)

        if not np.any(finite):
            raise ValueError(f"{cube_id}: spectral channel {k + 1} contains no finite pixels.")

        if not np.all(finite):
            median = np.median(image[finite])
            n_replaced += np.count_nonzero(~finite)
            image[~finite] = median

    logger.info("%s | Replaced %d non-finite pixels in diagnostic cube.", cube_id, n_replaced)

    return data


def run_slice_offset_diagnostic(cube_info, npoints=20, method=1, output_dir=None, plots=False):
    """Run Numina slice-offset diagnostics without modifying the original FITS cube."""

    executable = shutil.which("numina-measure_slice_xy_offsets_in_3d_cube")

    if executable is None:
        raise RuntimeError("Numina slice-offset executable not found.")

    cube_id = cube_info["id"]
    filename = Path(cube_info["path"]).resolve()
    wcs_ext = cube_info["wcs_ext"]
    output_dir = Path(output_dir or Path.cwd()).resolve() / f"slice_offsets_{cube_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    working_file = output_dir / f"{filename.stem}_slice_diagnostic.fits"
    log_file = output_dir / "numina_execution.log"

    # Create an independent primary-HDU cube
    with fits.open(filename, memmap=True) as hdul:
        hdu = hdul[wcs_ext]

        if hdu.data is None or hdu.data.ndim != 3:
            raise ValueError(f"{cube_id}: selected extension does not contain a 3D cube.")

        nlam = hdu.data.shape[0]

        if not 2 <= npoints <= nlam:
            raise ValueError(f"{cube_id}: npoints must be between 2 and {nlam}.")

        #header = hdu.header.copy()
        #data = hdu.data.copy()

        header = hdu.header.copy()
        data = np.asarray(hdu.data, dtype=float).copy()

        # Inspect and sanitize the diagnostic copy
        inspect_cube_slices(data, cube_id)
        data = prepare_cube_for_slice_diagnostic(data, cube_id)

    # Prevent any accidental modification of the input
    if working_file.resolve() == filename:
        raise ValueError("Diagnostic file cannot overwrite the original cube.")

    fits.PrimaryHDU(data=data, header=header).writeto(working_file, overwrite=True)

    # The result extension is explicitly named
    command = [executable, str(working_file), str(npoints), "--extname", "ADRCROSS", "--method", str(method)]

    if plots:
        command.append("--plots")

    if logger.isEnabledFor(logging.DEBUG):
        command.extend(["--log-level", "DEBUG"])

    logger.info("%s | Running Numina slice-offset diagnostic.", cube_id)
    logger.debug("Command: %s", " ".join(command))

    result = subprocess.run(command, cwd=output_dir, capture_output=True, text=True, check=False)

    log_file.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8"
    )

    if result.returncode != 0:
        details = "\n".join((result.stderr or result.stdout or "No output.").splitlines()[-30:])
        logger.error("%s | Numina diagnostic failed:\n%s", cube_id, details)
        raise RuntimeError(f"{cube_id}: Numina failed with exit code {result.returncode}. See {log_file}")

    logger.info("%s | Numina diagnostic completed: %s", cube_id, working_file)

    return output_dir





# Execution in all the cubes
# The diagnostic works on the original cubes, it doesn't matter if later we will use tables or the same cubes


def run_all_slice_offset_diagnostics(config, npoints=20, method=1, plots=False):
    """Run slice-offset diagnostics independently for each cube."""

    results = {}

    for cube_info in config["cubes"]:
        cube_id = cube_info["id"]

        logger.info("Measuring wavelength-dependent offsets for %s", cube_id)

        try:
            output_dir = run_slice_offset_diagnostic(cube_info, npoints=npoints, method=method, plots=plots)
            results[cube_id] = {"status": "success", "output_dir": str(output_dir)}

        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("%s | Slice-offset diagnostic failed: %s", cube_id, exc)
            results[cube_id] = {"status": "failed", "error": str(exc)}

    n_success = sum(result["status"] == "success" for result in results.values())
    logger.info("Slice-offset diagnostics completed: %d/%d successful.", n_success, len(results))

    return results

