#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of cubealign.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE
#

import logging
import matplotlib.pyplot as plt
import numpy as np

from ..save_cubes.validate_wcs import validate_wcs_for_crpix_update
from ..save_cubes.calculate_crpix_corrections import calculate_crpix_corrections
from ..save_cubes.write_cubes import write_aligned_tables, write_aligned_cubes

logger = logging.getLogger(__name__)


def save_alignment_results(config, images, wcs_dict, reference_wcs, offsets, astrometry_result, history_line_command):
    """Validate WCS compatibility and write corrected FITS cubes and tables."""

    ref_id = config["ref_cube"]
    ref_shape = images[ref_id].shape

    # Validate WCS compatibility
    for cube_info in config["cubes"]:
        cube_id = cube_info["id"]

        if images[cube_id].shape != ref_shape:
            raise ValueError(f"{cube_id}: spatial shape differs from reference. Reprojection required.")

        if cube_id != ref_id:
            validate_wcs_for_crpix_update(reference_wcs, wcs_dict[cube_id], ref_shape)

    logger.info("WCS compatibility verified.")

    # Combine relative and external offsets
    corrections = calculate_crpix_corrections(config, offsets, astrometry_result)

    # Write corrected cubes
    written_cubes = write_aligned_cubes(config, corrections, offsets, astrometry_result, history_line_command)

    # Write corrected FITS tables when possible
    written_tables = write_aligned_tables(config, corrections, wcs_dict)

    logger.info("Corrected cubes: %d", len(written_cubes))
    logger.info("Corrected tables: %d", len(written_tables))

    return corrections, written_cubes, written_tables
