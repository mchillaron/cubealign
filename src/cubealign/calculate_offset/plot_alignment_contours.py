#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

from matplotlib.lines import Line2D
from numina.tools.measure_xy_offset_2dimages import measure_xy_offset_2dimages
from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np
import re


from ..cube_table_to_image.plot_line_images import add_top_colorbar, output_figure

logger = logging.getLogger(__name__)

def get_contour_levels(image, percentiles=(70, 80, 90, 95)):
    """Calculate positive intensity contour levels."""

    valid = image[np.isfinite(image) & (image > 0)]

    if valid.size < 4:
        return np.array([])

    levels = np.unique(np.percentile(valid, percentiles))
    return levels if len(levels) > 0 else np.array([])




def plot_offset_comparison(reference, moving, results, cube_id, ref_id, pmin=1.0, pmax=99.5, cmap="plasma", show=False):
    """Compare original and aligned contours for the available methods."""

    method_results = results["methods"]
    methods = list(method_results)
    panels = [("Original", moving, results["original_pearson"])]

    for name in methods:
        panels.append((name.upper(), method_results[name]["aligned_image"], method_results[name]["pearson"]))

    finite = reference[np.isfinite(reference)]

    if finite.size == 0:
        raise ValueError("Reference image contains no finite pixels.")

    vmin, vmax = np.percentile(finite, [pmin, pmax])

    if vmin == vmax:
        vmin, vmax = finite.min() - 0.5, finite.max() + 0.5

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
    axes = axes.ravel()

    for ax, (name, image, pearson) in zip(axes, panels):
        ax.imshow(reference, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

        levels = get_contour_levels(image)

        if len(levels) > 0:
            ax.contour(image, levels=levels, colors="cyan", linewidths=1.5, linestyles="--")

        ax.set_title(f"{cube_id} - {name}\nPearson = {pearson:.4f}")
        ax.set_xlabel("X [pix]")
        ax.set_ylabel("Y [pix]")

    fig.suptitle(f"Alignment of {cube_id} relative to {ref_id}", fontsize=14)
    fig.tight_layout()

    #filename = Path.cwd() / f"offset_comparison_{cube_id}.png"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cube_id))
    filename = Path.cwd() / f"offset_comparison_{safe_id}.png"

    fig.savefig(filename, dpi=200, bbox_inches="tight")
    logger.info("Comparison figure saved: %s", filename)

    if show:
        plt.show(block=False)
        plt.pause(0.3)
    else:
        plt.close(fig)

    return fig if show else None



def plot_final_alignment(images, aligned_images, cube_info, ref_id, cmap="magma_r", pmin=1.0, pmax=99.5):
    """Plot the reference image with contours from all aligned images."""

    reference = images[ref_id]
    finite = reference[np.isfinite(reference)]

    if finite.size == 0:
        raise ValueError("Reference image contains no finite pixels.")

    vmin, vmax = np.percentile(finite, [pmin, pmax])

    if vmin == vmax:
        vmin, vmax = finite.min() - 0.5, finite.max() + 0.5

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(reference, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    colors = plt.get_cmap("tab10")
    linestyles = ["--", ":", "-.", "-"]
    legend_elements = []

    moving_cubes = [info for info in cube_info if info["id"] != ref_id]

    for i, info in enumerate(moving_cubes):
        cube_id = info["id"]
        image = aligned_images[cube_id]
        levels = get_contour_levels(image)

        if len(levels) == 0:
            logger.warning("%s: no valid contour levels.", cube_id)
            continue

        color = colors(i % 10)
        linestyle = linestyles[i % len(linestyles)]
        label = f"{cube_id} - {info.get('line', 'Unknown')}"

        ax.contour(image, levels=levels, colors=[color], linewidths=1.6, linestyles=linestyle)
        legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.6, label=label))

    if legend_elements:
        ax.legend(handles=legend_elements, loc="best", fontsize=9)

    ax.set_xlabel("X [pix]")
    ax.set_ylabel("Y [pix]")
    ax.set_title(f"Final emission-line registration - Reference: {ref_id}")

    add_top_colorbar(fig, ax, im, label="Reference image intensity")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_figure(fig, "final_alignment_contours.png")

