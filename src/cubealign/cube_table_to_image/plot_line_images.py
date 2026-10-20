#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from astropy.visualization import ZScaleInterval
from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def output_figure(fig, filename):
    """Save figures at INFO level and additionally display them at DEBUG level."""

    if logger.isEnabledFor(logging.INFO):
        output_path = Path.cwd() / filename
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info("Figure saved: %s", output_path)

    if logger.isEnabledFor(logging.DEBUG):
        plt.show()

    plt.close(fig)


def get_zscale_limits(image, contrast=0.25):
    """Return ZScale display limits for finite image pixels."""

    finite = image[np.isfinite(image)]

    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")

    if np.ptp(finite) == 0:
        return finite[0] - 0.5, finite[0] + 0.5

    return ZScaleInterval(contrast=contrast).get_limits(finite)


def add_top_colorbar(fig, ax, im, label=""):
    """Attach a horizontal colorbar above an image."""

    cax = ax.inset_axes([0.0, 1.02, 1.0, 0.045])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    return cbar


def plot_alignment_images(images, cube_info, mode="cubes", cmap="plasma", pmin=1.0, pmax=99.5, vmin=None, vmax=None):
    """Plot alignment images with independent percentile-based intensity limits."""

    if not logger.isEnabledFor(logging.INFO):
        return

    if not 0 <= pmin < pmax <= 100:
        raise ValueError("Percentiles must satisfy 0 <= pmin < pmax <= 100.")

    n = len(cube_info)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes.ravel()

    for ax, info in zip(axes, cube_info):
        cube_id = info["id"]
        line_name = info.get("line", "Unknown line")
        image = images[cube_id]

        finite = image[np.isfinite(image)]

        if finite.size == 0:
            raise ValueError(f"{cube_id}: image contains no finite pixels.")

        # Independent percentile limits for each image
        auto_min, auto_max = np.percentile(finite, [pmin, pmax])
        image_min = auto_min if vmin is None else vmin
        image_max = auto_max if vmax is None else vmax

        if image_min == image_max:
            image_min -= 0.5
            image_max += 0.5

        if image_min > image_max:
            raise ValueError(f"{cube_id}: vmin must be smaller than vmax.")

        logger.debug("%s | vmin = %.6g | vmax = %.6g | Percentiles: %.1f-%.1f", cube_id, image_min, image_max, pmin, pmax)

        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=image_min, vmax=image_max)

        ax.set_title(f"{cube_id} - {line_name}")
        ax.set_xlabel("X [pix]")
        ax.set_ylabel("Y [pix]")

        label = "Line flux" if mode == "tables" else "Collapsed signal"
        add_top_colorbar(fig, ax, im, label=label)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_figure(fig, f"alignment_images_{mode}.png")


