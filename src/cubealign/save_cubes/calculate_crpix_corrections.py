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

logger = logging.getLogger(__name__)

def calculate_crpix_corrections(config, offsets, astrometry_result=None):
    """Combine internal alignment and external astrometric offsets."""

    ref_id = config["ref_cube"]
    dx_ast = astrometry_result["dx"] if astrometry_result is not None else 0.0
    dy_ast = astrometry_result["dy"] if astrometry_result is not None else 0.0

    corrections = {}

    for cube in config["cubes"]:
        cube_id = cube["id"]
        dx_rel = offsets[cube_id]["dx"] if cube_id != ref_id else 0.0
        dy_rel = offsets[cube_id]["dy"] if cube_id != ref_id else 0.0

        corrections[cube_id] = {
            "dx_relative": dx_rel,
            "dy_relative": dy_rel,
            "dx_astrometry": dx_ast,
            "dy_astrometry": dy_ast,
            "dcrpix1": dx_rel - dx_ast,
            "dcrpix2": dy_rel - dy_ast
        }

        logger.info("%s | dCRPIX1 = %+.4f | dCRPIX2 = %+.4f", cube_id,
                    corrections[cube_id]["dcrpix1"], corrections[cube_id]["dcrpix2"])

    return corrections
