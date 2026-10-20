# CubeAlign

**Spatial alignment and astrometric refinement of astronomical IFU datacubes.**

CubeAlign is a Python command-line tool for measuring and correcting spatial offsets between integral-field spectroscopy (IFS/IFU) datacubes.

The tool compares two-dimensional emission-line maps extracted from cubes or reconstructed from fitted-line FITS tables. A selected cube serves as the reference. Optionally, CubeAlign refines its astrometry against a FITS image from an external survey (for example, SDSS or Pan-STARRS), using a WCS-based cutout and reprojection.

CubeAlign writes new FITS products with updated spatial WCS reference pixels (`CRPIX1`, `CRPIX2`) and provenance entries in `HISTORY`.

## Features

- Align any number of input cubes to a user-selected reference cube.
- Read cube paths, per-cube WCS extensions, line labels, and spectral intervals from a commented YAML file.
- Two input modes:
  - **Cube mode:** collapse a specified rest-frame wavelength interval for each cube, accounting for a shared galaxy redshift.
  - **Table mode:** reconstruct maps from fitted-line FITS tables using configurable X, Y, and flux column names.
- Measure offsets with the two methods provided by Numina (`m1` and `m2`).
- Assess candidate solutions using Pearson correlation and comparison plots.
- Choose a method interactively in `DEBUG`, or select the higher valid Pearson correlation automatically with `--method both` at `INFO` level.
- Optionally align the reference cube astrometrically to an external FITS image using an automatic cutout and reprojection onto the reference cube's pixel grid.
- Save diagnostic plots and corrected FITS copies; record the command and alignment metadata in FITS `HISTORY`.

## Quick start

Run relative alignment:

```bash
cube-align alignment_config.yaml --method both --log-level INFO
```

To inspect and select each method interactively:

```bash
cube-align alignment_config.yaml --method both --log-level DEBUG
```

To run a previously selected method without prompting:

```bash
cube-align alignment_config.yaml --method m1 --log-level INFO
```

Paths inside the YAML are resolved relative to the YAML file's directory. Diagnostic plots and corrected FITS products are saved in the **current working directory** by default.


## Command-line options
    
| Argument | Purpose | Default |
| --- | --- | --- |
| `alignment_config.yaml` | Required YAML configuration file | Required |
| `--wcs-ext` | Default FITS extension containing cube WCS; overridden per cube | `0` |
| `--method {m1,m2,both}` | Numina offset method(s) | `both` |
| `--refine-astrometry` | Enable external astrometric refinement | Disabled |
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | Logging and plot-display behavior | `INFO` |

With `--method both`, the implementation selects a method **independently for each moving cube**. At `DEBUG` level, comparison figures are displayed and the user can make a choice; at `INFO` level, the program selects the method with the higher eligible Pearson coefficient. Specifying `m1` or `m2` skips that choice.

