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
from astropy.visualization import PercentileInterval

from .read_cubes import wavelength_axis
from .plot_line_images import output_figure

logger = logging.getLogger(__name__)

def collapse_emission_line(cube, wcs, wave_min, wave_max, cube_id=None):
    """Collapse a datacube over an observed-frame wavelength interval."""

    if not np.isfinite(wave_min) or not np.isfinite(wave_max) or wave_min >= wave_max:
        raise ValueError(f"{cube_id}: invalid wavelength interval.")

    lam = wavelength_axis(wcs, cube.shape[0])
    mask = (lam >= wave_min) & (lam <= wave_max)

    if not np.any(mask):
        raise ValueError(f"{cube_id}: no spectral pixels within {wave_min:.2f}-{wave_max:.2f} Angstrom.")

    selected = cube[mask, :, :]
    image = np.nansum(selected, axis=0)
    image[np.all(~np.isfinite(selected), axis=0)] = np.nan

    logger.debug("%s | Integration: %.2f - %.2f Angstrom | Channels: %d", cube_id, wave_min, wave_max, np.count_nonzero(mask))

    return image



def plot_collapsed_images(images, cubes, cmap="magma_r", percentile=99.5):
    """Display collapsed emission-line images for all input cubes."""

    n = len(cubes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes.ravel()

    for ax, cube_info in zip(axes, cubes):
        cube_id = cube_info["id"]
        image = images[cube_id]
        line_name = cube_info.get("line", "Collapsed image")

        valid = image[np.isfinite(image)]

        if valid.size == 0:
            raise ValueError(f"{cube_id}: collapsed image contains no finite pixels.")

        if np.ptp(valid) == 0:
            vmin, vmax = valid[0] - 0.5, valid[0] + 0.5
        else:
            vmin, vmax = PercentileInterval(percentile).get_limits(valid)

        ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{cube_id} - {line_name}")
        ax.set_xlabel("X [pix]")
        ax.set_ylabel("Y [pix]")

    fig.tight_layout()
    output_figure(fig, "collapsed_cube_lines.png")
    
