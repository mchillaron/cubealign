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
from rich.console import Console
from pathlib import Path

import logging
import numpy as np
import sys
import yaml

console = Console()
logger = logging.getLogger(__name__)

class ConfigurationError(ValueError):
    """Exception raised when the alignment configuration is invalid."""
    pass


def fits_ext(value):
    """Convert a FITS extension identifier to an integer or EXTNAME."""

    if isinstance(value, bool):
        raise ConfigurationError(f"Invalid FITS extension: {value!r}")

    if isinstance(value, int):
        if value < 0:
            raise ConfigurationError("FITS extension index must be >= 0.")
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ConfigurationError("FITS extension cannot be empty.")
        if value.isdecimal():
            return int(value)
        return value

    raise ConfigurationError(f"Invalid FITS extension: {value!r}")



def load_config(path, wcs_ext, refine_astrometry=False):
    """Load and validate the YAML configuration for cube alignment."""

    def resolve_path(value, root, description):
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{description}: missing or invalid path.")

        filepath = Path(value).expanduser()
        filepath = (root / filepath).resolve()

        if not filepath.is_file():
            raise ConfigurationError(f"{description}: file not found: {filepath}")

        return filepath

    def verify_fits(path, ext, description, table=False, columns=None, check_wcs=False):
        try:
            with fits.open(path, memmap=True) as hdul:
                hdu = hdul[ext]

                if table:
                    if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                        raise ConfigurationError(f"{description}: extension {ext} is not a FITS table.")

                    missing = set(columns.values()) - set(hdu.columns.names)
                    if missing:
                        raise ConfigurationError(f"{description}: missing columns {sorted(missing)}.")

                else:
                    if hdu.data is None or hdu.data.ndim < 2:
                        raise ConfigurationError(f"{description}: extension {ext} has no valid image data.")

                if check_wcs and not WCS(hdu.header).has_celestial:
                    raise ConfigurationError(f"{description}: no celestial WCS found in extension {ext}.")

        except (OSError, KeyError, IndexError, ValueError, TypeError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"{description}: invalid FITS file or extension: {exc}") from exc

    # Read YAML configuration
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read YAML configuration: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigurationError("YAML root must be a dictionary.")

    root = config_path.parent
    wcs_ext = fits_ext(wcs_ext)

    # Validate redshift
    try:
        redshift = float(config["redshift"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("Missing or invalid galaxy redshift.") from exc

    if not np.isfinite(redshift) or redshift <= -1:
        raise ConfigurationError("Galaxy redshift must be finite and greater than -1.")

    config["redshift"] = redshift

    # Validate cubes

    cubes = config.get("cubes")

    if not isinstance(cubes, list) or len(cubes) < 2:
        raise ConfigurationError("At least two cubes must be provided.")

    cube_ids = set()
    cube_paths = set()

    for i, cube in enumerate(cubes):
        if not isinstance(cube, dict):
            raise ConfigurationError(f"Cube entry {i} must be a dictionary.")

        cube_id = cube.get("id")
        if not isinstance(cube_id, str) or not cube_id.strip():
            raise ConfigurationError(f"Cube entry {i}: invalid or missing ID.")

        if cube_id in cube_ids:
            raise ConfigurationError(f"Duplicated cube ID: {cube_id}")

        cube_ids.add(cube_id)
        cube["path"] = resolve_path(cube.get("path"), root, f"Cube {cube_id}")

        if cube["path"] in cube_paths:
            raise ConfigurationError(f"Duplicated cube file: {cube['path']}")

        cube_paths.add(cube["path"])

        # Cube-specific WCS extension (fallback to CLI default)
        cube["wcs_ext"] = fits_ext(cube.get("wcs_ext", wcs_ext))

        verify_fits(cube["path"], cube["wcs_ext"], f"Cube {cube_id}", check_wcs=True)

        if "line" in cube and (not isinstance(cube["line"], str) or not cube["line"].strip()):
            raise ConfigurationError(f"Cube {cube_id}: invalid line name.")

    # Reference cube

    ref_cube = config.get("ref_cube")

    if ref_cube not in cube_ids:
        raise ConfigurationError(f"Reference cube '{ref_cube}' is not present in the cube list.")

    # Determine alignment mode
    tables = config.get("tables")
    if tables is None:
        tables = []

    if not isinstance(tables, list):
        raise ConfigurationError("'tables' must be a list.")

    mode = "tables" if tables else "cubes"

    # Table mode
    if mode == "tables":
        col_names = config.get("col_names")
        required = ("x_coord", "y_coord", "line_flux")

        if not isinstance(col_names, dict):
            raise ConfigurationError("'col_names' must be a dictionary.")

        for col in required:
            if not isinstance(col_names.get(col), str) or not col_names[col].strip():
                raise ConfigurationError(f"Missing or invalid column name: {col}")

        table_ids = set()

        for i, table in enumerate(tables):
            if not isinstance(table, dict):
                raise ConfigurationError(f"Table entry {i} must be a dictionary.")

            cube_id = table.get("cube_id")

            if cube_id not in cube_ids:
                raise ConfigurationError(f"Table {i}: unknown cube ID '{cube_id}'.")

            if cube_id in table_ids:
                raise ConfigurationError(f"Duplicated table for cube '{cube_id}'.")

            table_ids.add(cube_id)
            table["path"] = resolve_path(table.get("path"), root, f"Table {cube_id}")
            table["ext"] = fits_ext(table.get("ext", 1))

            verify_fits(table["path"], table["ext"], f"Table {cube_id}", table=True, columns=col_names)

        if table_ids != cube_ids:
            raise ConfigurationError(f"Missing tables for cubes: {sorted(cube_ids - table_ids)}")

        for cube in cubes:
            if not cube.get("line"):
                raise ConfigurationError(f"Cube {cube['id']}: line name required in table mode.")

    # Cube mode: validate spectral collapse intervals
    else:
        for cube in cubes:
            cube_id = cube["id"]
            collapse = cube.get("collapse")

            if not isinstance(collapse, dict):
                raise ConfigurationError(f"Cube {cube_id}: missing collapse interval.")

            try:
                wave_min = float(collapse["min"])
                wave_max = float(collapse["max"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ConfigurationError(f"Cube {cube_id}: invalid collapse limits.") from exc

            if not np.isfinite(wave_min) or not np.isfinite(wave_max) or wave_min <= 0 or wave_min >= wave_max:
                raise ConfigurationError(f"Cube {cube_id}: invalid rest-frame wavelength interval.")

            # Rest-frame wavelength interval
            collapse["rest_min"] = wave_min
            collapse["rest_max"] = wave_max

            # Observed-frame wavelength interval
            collapse["obs_min"] = wave_min * (1.0 + redshift)
            collapse["obs_max"] = wave_max * (1.0 + redshift)


    config["fits_coordinates"] = config.get("fits_coordinates", True)

    if not isinstance(config["fits_coordinates"], bool):
        raise ConfigurationError("'fits_coordinates' must be true or false.")


    # External astrometric reference
    astrometry = config.get("astrometry")
    if astrometry is None:
        astrometry = {}

    if not isinstance(astrometry, dict):
        raise ConfigurationError("'astrometry' must be a dictionary.")

    if refine_astrometry:
        astrometry["image"] = resolve_path(astrometry.get("image"), root, "Astrometric reference")
        astrometry["image_ext"] = fits_ext(astrometry.get("image_ext", 0))
        verify_fits(astrometry["image"], astrometry["image_ext"], "Astrometric reference", check_wcs=True)

    cutout = astrometry.get("cutout", {})

    if not isinstance(cutout, dict):
        raise ConfigurationError("'astrometry.cutout' must be a dictionary.")

    try:
        margin = float(cutout.get("margin", 1.5))
        padding = float(cutout.get("padding_arcsec", 5.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("Invalid astrometry cutout parameters.") from exc

    if not np.isfinite(margin) or not np.isfinite(padding) or margin < 1 or padding < 0:
        raise ConfigurationError("Astrometry cutout requires margin >= 1 and padding_arcsec >= 0.")

    astrometry["cutout"] = {"margin": margin, "padding_arcsec": padding}

    # Store validated configuration
    config["mode"] = mode
    config["wcs_ext"] = wcs_ext
    config["astrometry"] = astrometry

    return config



def choose_method_old():
    """Interactively select the preferred alignment method."""

    if not sys.stdin.isatty():
        raise ConfigurationError("Interactive selection requires a terminal. Use --method m1 or --method m2.")

    while True:
        choice = console.input("\n[bold cyan]Select alignment method [m1/m2] (q to quit): [/bold cyan]").strip().lower()

        if choice in ("m1", "m2"):
            console.print(f"[green]Selected method: {choice}[/green]")
            return choice

        if choice in ("q", "quit"):
            raise KeyboardInterrupt

        console.print("[yellow]Invalid selection. Enter m1, m2 or q.[/yellow]")


def choose_method(results, interactive=False):
    """Choose the alignment method manually or by maximum Pearson correlation."""

    methods = results["methods"]
    valid = {name: data["pearson"] for name, data in methods.items() if np.isfinite(data["pearson"])}

    if not valid:
        raise ValueError("No valid Pearson correlation available for method selection.")

    best = max(valid, key=valid.get)

    if not interactive or len(methods) == 1:
        logger.info("Selected method: %s (Pearson = %.5f)", best.upper(), valid[best])
        return best

    if not sys.stdin.isatty():
        raise RuntimeError("Interactive selection requires a terminal.")

    console.print(f"\n[cyan]Automatic recommendation: {best.upper()} (Pearson = {valid[best]:.5f})[/cyan]")

    while True:
        choice = console.input("[bold]Select method [m1/m2] (Enter = recommended, q = quit): [/bold]").strip().lower()

        if not choice:
            return best

        if choice in methods:
            logger.info("User selected method: %s", choice.upper())
            return choice

        if choice in ("q", "quit"):
            raise KeyboardInterrupt

        console.print("[yellow]Invalid selection.[/yellow]")

