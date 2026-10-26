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

def normalize_alignment_image(image, cube_id=None):
    """Normalize an alignment image by its maximum finite positive intensity."""
    ## ATTENTION: If cosmic rays are present, they can dominate for now the normalization !!

    data = np.asarray(image, dtype=float).copy()
    finite = np.isfinite(data)

    if not np.any(finite):
        raise ValueError(f"{cube_id}: image contains no finite pixels.")

    maximum = np.max(data[finite])

    if maximum <= 0:
        raise ValueError(f"{cube_id}: normalization requires a positive image maximum.")

    data[finite] /= maximum

    logger.debug("%s | Normalization maximum: %.6g", cube_id, maximum)

    return data
