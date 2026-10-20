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

from rich.console import Console
from rich.logging import RichHandler

import argparse
import logging
import matplotlib.pyplot as plt
import sys
import shlex
import time

from .initial_functions import load_config, choose_method, fits_ext
from .cube_table_to_image.read_cubes import read_cube, pixel_scales_arcsec, log_wcs_info
from .cube_table_to_image.read_tables import read_line_table, inspect_table_coordinates, inspect_table_header, table_to_image
from .cube_table_to_image.collapse_cube import collapse_emission_line, plot_collapsed_images
from .cube_table_to_image.plot_line_images import plot_alignment_images
from .calculate_offset.calculate_offset_base import calculate_offset_base
from .check_astrometry.check_astrometry_base import refine_reference_astrometry
from .check_astrometry.plot_astrometry_images import plot_final_astrometry
from .save_cubes.save_alignment_results import save_alignment_results

console = Console()
logger = logging.getLogger(__name__)

class ConfigurationError(ValueError):
    """Exception raised when the alignment configuration is invalid."""
    pass

def measure_cube_alignment(config, method, 
                            refine_astrometry,
                            choose_method,
                            history_line_command, 
                            time_start):
    
    """Measure relative spatial alignment of IFU datacubes."""

    logger.info("Starting cube alignment.")

    # Read cubes and inspect WCS information
    cubes = {}
    headers = {}
    wcs_dict = {}
    pixel_scales = {}
    images = {}
    tables = {}

    for cube_info in config["cubes"]:
        cube_id = cube_info["id"]
        filename = cube_info["path"]
        wcs_ext = cube_info["wcs_ext"]

        logger.info("Reading cube: %s", cube_id)

        cube, header, wcs = read_cube(filename, wcs_ext)

        cubes[cube_id] = cube
        headers[cube_id] = header
        wcs_dict[cube_id] = wcs
        pixel_scales[cube_id] = pixel_scales_arcsec(wcs, cube_id)

        logger.debug("%s | Shape: %s", cube_id, cube.shape)
        log_wcs_info(header, cube_id)

    # Cube-mode: Generate collapsed emission-line images
    if config["mode"] == "cubes":
        logger.info("Generating collapsed emission-line images.")

        for cube_info in config["cubes"]:
            cube_id = cube_info["id"]
            collapse = cube_info["collapse"]

            images[cube_id] = collapse_emission_line(
                cubes[cube_id], wcs_dict[cube_id],
                collapse["obs_min"], collapse["obs_max"], cube_id
            )

        plot_alignment_images(images, config["cubes"], mode="cubes", cmap="plasma")

    # Table-mode: reconstruct fitted-line maps
    elif config["mode"] == "tables":
        logger.info("Generating fitted emission-line maps from FITS tables.")

        col_names = config["col_names"]
        fits_coordinates = config["fits_coordinates"]

        for table_info in config["tables"]:
            cube_id = table_info["cube_id"]
            filename = table_info["path"]
            table_ext = table_info["ext"]

            logger.info("Processing fitted-line table: %s", cube_id)

            table = read_line_table(filename, table_ext, cube_id)
            tables[cube_id] = table

            inspect_table_coordinates(
                table, cubes[cube_id].shape[1:],
                col_names, cube_id, fits_coordinates
            )

            inspect_table_header(filename, cube_id)

            images[cube_id] = table_to_image(
                table, cubes[cube_id].shape[1:],
                col_names, cube_id, fits_coordinates
            )

        plot_alignment_images(images, config["cubes"], mode="tables", cmap="plasma")

    else:
        raise ConfigurationError(f"Unknown alignment mode: {config['mode']}")


    # Offsets measurements
    logger.info("Relative cube alignment starts") 
    reference_image, reference_wcs, alignment_results, offsets, aligned_images = calculate_offset_base(config, images, wcs_dict, method)
    logger.info("Relative cube alignment completed.")

    # Optional external astrometric refinement
    astrometry_result = None

    if refine_astrometry:
        logger.info("External astrometric refinement starts.")

        astrometry_result = refine_reference_astrometry(config, reference_image, reference_wcs, method, choose_method)
        plot_final_astrometry(reference_image, astrometry_result, config["ref_cube"],label=config["astrometry"].get("label", "External image"))
        logger.info("External astrometric refinement completed.")

    else:
        logger.info("External astrometric refinement disabled.")

    # Write corrected FITS products
    logger.info("Writing corrected FITS products.")
    corrections, written_cubes, written_tables = save_alignment_results(
        config, images, wcs_dict, reference_wcs, offsets,
        astrometry_result, history_line_command
    )

    # Final summary
    logger.info("=" * 60)
    logger.info("Cube alignment completed successfully.")
    logger.info("Corrected cubes: %d", len(written_cubes))
    logger.info("Corrected tables: %d", len(written_tables))
    logger.info("=" * 60)

    return {
        "cubes": cubes,
        "headers": headers,
        "wcs": wcs_dict,
        "pixel_scales": pixel_scales,
        "tables": tables,
        "images": images,
        "reference_image": reference_image,
        "reference_wcs": reference_wcs,
        "alignment_results": alignment_results,
        "offsets": offsets,
        "aligned_images": aligned_images,
        "alignment_results": alignment_results,
        "aligned_images": aligned_images,
        "astrometry": astrometry_result,
        "corrections": corrections,
        "written_cubes": written_cubes,
        "written_tables": written_tables
    }



def main():
    start = time.perf_counter()

    parser = argparse.ArgumentParser(description=("Spatial alignment of IFU datacubes using collapsed spectral images or fitted-line flux tables, with optional external astrometric refinement."),
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--wcs-ext", type=str, default="0", help=("Default FITS extension containing the WCS, overridden by cube-specific wcs_ext values in the YAML."),)
    parser.add_argument('-m', '--method', type=str, choices=["m1", "m2", "both"], default="both", help=("Alignment method: m1, m2, or both. When both is selected, the user chooses the preferred solution interactively."),)
    parser.add_argument("--refine-astrometry", action="store_true", default=False, help=("Refine the absolute astrometry using an external reference FITS image specified in the YAML file."),)
    parser.add_argument("--log-level", type=str.upper, choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL",],default="INFO",help="Logging verbosity level.",)

    args = parser.parse_args()

    # Logging configuration
    if args.log_level in ["DEBUG","INFO","WARNING","ERROR","CRITICAL"]:
        handlers = [
            RichHandler(
                console=console,
                show_time=False,
                show_level=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
            )
        ]
    else:
        handlers = [
            RichHandler(
                console=console,
                show_time=False,
                show_level=False,
                show_path=False,
                markup=True,
            )
        ]

    logging.basicConfig(level=args.log_level, format="%(message)s", handlers=handlers, force=True,)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


    # Save the executed command for FITS HISTORY
    history_line_command = shlex.join(sys.argv)
    logger.debug("Executed command: %s", history_line_command,)

    # Load and validate configuration
    try:
        config = load_config(args.config, fits_ext(args.wcs_ext), args.refine_astrometry,)

        logger.info("Input configuration: %s", args.config,)
        logger.info("Number of cubes: %d", len(config["cubes"]),)
        logger.info("Reference cube: %s", config["ref_cube"],)
        logger.info("Alignment mode: %s", config["mode"],)
        logger.info("Alignment method: %s", args.method,)
        logger.info("External astrometric refinement: %s", args.refine_astrometry,)

    
        # Run cube alignment
        measure_cube_alignment(config=config, 
                               method=args.method, 
                               refine_astrometry=args.refine_astrometry,
                               choose_method=(choose_method if args.method == "both" else None),
                               history_line_command=history_line_command, time_start=start)

        logger.info("Cube alignment completed successfully.")
        return 0

    except ConfigurationError as exc:

        logger.error("Configuration error: %s", exc,)
        return 2

    except (OSError, ValueError, ImportError) as exc:
        logger.error("Alignment failed: %s", exc,)
        return 1

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by the user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
