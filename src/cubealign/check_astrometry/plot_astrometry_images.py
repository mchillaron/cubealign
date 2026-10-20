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
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.lines import Line2D
from pathlib import Path
from reproject import reproject_interp

import logging
import matplotlib.pyplot as plt
import numpy as np

from ..cube_table_to_image.plot_line_images import output_figure, add_top_colorbar
from ..calculate_offset.plot_alignment_contours import get_contour_levels

logger = logging.getLogger(__name__)


def plot_astrometry_images(reference, external, ref_id, label="External image", pmin=1, pmax=99.5):
    """Compare the reference cube image and the reprojected external image."""

    if not logger.isEnabledFor(logging.INFO):
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, image, title, cmap in zip(
        axes, (reference, external), (f"{ref_id} - Reference", label), ("plasma", "viridis")
    ):
        finite = image[np.isfinite(image)]

        if finite.size == 0:
            raise ValueError(f"{title}: no finite pixels.")

        vmin, vmax = np.percentile(finite, [pmin, pmax])
        if vmin == vmax:
            vmin, vmax = finite.min() - 0.5, finite.max() + 0.5

        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X [reference pix]")
        ax.set_ylabel("Y [reference pix]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_figure(fig, "astrometry_reference_comparison.png")



def plot_final_astrometry(reference_image, astrometry_result, ref_id, label="External image", pmin=1, pmax=99.5):
    """Plot selected external astrometric contours over the reference image."""

    if not logger.isEnabledFor(logging.INFO):
        return

    finite = reference_image[np.isfinite(reference_image)]
    vmin, vmax = np.percentile(finite, [pmin, pmax])

    if vmin == vmax:
        vmin, vmax = finite.min() - 0.5, finite.max() + 0.5

    aligned_external = astrometry_result["aligned_image"]
    levels = get_contour_levels(aligned_external)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(reference_image, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)

    if len(levels):
        ax.contour(aligned_external, levels=levels, colors="cyan", linewidths=1.6, linestyles="--")
        ax.legend(handles=[Line2D([0], [0], color="cyan", linestyle="--", label=label)], loc="best")

    ax.set_xlabel("X [reference pix]")
    ax.set_ylabel("Y [reference pix]")
    ax.set_title(f"Selected astrometric registration - {ref_id}")
    add_top_colorbar(fig, ax, im, label="Reference image intensity")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_figure(fig, "final_astrometric_alignment.png")
