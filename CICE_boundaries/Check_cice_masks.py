#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare the target CICE kmt mask with tmask from the target reference file.
"""

import netCDF4
import numpy as np


KMT_FILE = "/cluster/shared/arcticfjord/input_data/s800_cice_processed_a4_mod/cice.kmt.nc"
TMASK_FILE = "iceh.2015-11.selvar.tmask.nc"


def read_array(ncfile, varname):
    """Read a NetCDF variable as float64, converting masked values to NaN."""
    with netCDF4.Dataset(ncfile, "r") as nc:
        var = nc.variables[varname]
        var.set_auto_mask(True)

        data = var[:]

    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)

    data = np.asarray(data, dtype=np.float64)

    # Replace common NetCDF/CICE large fill values.
    data[np.abs(data) > 1.0e20] = np.nan

    return data


def main():

    kmt = read_array(KMT_FILE, "kmt")
    tmask = read_array(TMASK_FILE, "tmask")

    print("Mask-file comparison")
    print("=" * 70)

    print("kmt shape:  ", kmt.shape)
    print("tmask shape:", tmask.shape)

    if kmt.shape != tmask.shape:
        raise ValueError(
            "Mask dimensions differ:\n"
            f"kmt   = {kmt.shape}\n"
            f"tmask = {tmask.shape}"
        )

    # Print actual values.
    print("\nkmt finite unique values:")
    print(np.unique(kmt[np.isfinite(kmt)]))

    print("\ntmask finite unique values:")
    print(np.unique(tmask[np.isfinite(tmask)]))

    # Assumed wet/dry interpretation.
    kmt_ocean = np.isfinite(kmt) & (kmt > 0.0)
    tmask_ocean = np.isfinite(tmask) & (tmask > 0.0)

    kmt_land = ~kmt_ocean
    tmask_land = ~tmask_ocean

    print("\nCell counts")
    print("-" * 70)
    print("Total cells:          ", kmt.size)

    print("kmt wet cells:        ", np.count_nonzero(kmt_ocean))
    print("kmt land cells:       ", np.count_nonzero(kmt_land))

    print("tmask wet cells:      ", np.count_nonzero(tmask_ocean))
    print("tmask land cells:     ", np.count_nonzero(tmask_land))

    # Direct comparison.
    different = kmt_ocean != tmask_ocean

    print("\nComparison")
    print("-" * 70)
    print(
        "Cells where wet/dry classification differs:",
        np.count_nonzero(different),
    )

    # Detailed breakdown.
    kmt_wet_tmask_land = kmt_ocean & ~tmask_ocean
    kmt_land_tmask_wet = ~kmt_ocean & tmask_ocean

    print(
        "kmt wet but tmask land:",
        np.count_nonzero(kmt_wet_tmask_land),
    )

    print(
        "kmt land but tmask wet:",
        np.count_nonzero(kmt_land_tmask_wet),
    )

    if not np.any(different):
        print("\nRESULT: kmt and tmask have identical wet/dry masks.")
        print("You can safely use:")
        print("")
        print("    target_ocean_mask = kmt > 0")
        print("")
        return

    # Locate differences.
    jj, ii = np.where(different)

    print("\nDifference index ranges")
    print("-" * 70)
    print("j range:", jj.min(), "to", jj.max())
    print("i range:", ii.min(), "to", ii.max())

    print("\nFirst 50 differing cells")
    print("-" * 70)
    print("      j       i           kmt         tmask"
          "       kmt_wet  tmask_wet")

    for j, i in list(zip(jj, ii))[:50]:
        print(
            f"{j:7d} {i:7d}"
            f" {kmt[j, i]:13.5g}"
            f" {tmask[j, i]:13.5g}"
            f" {str(bool(kmt_ocean[j, i])):>14s}"
            f" {str(bool(tmask_ocean[j, i])):>11s}"
        )

    # Identify whether differences occur only along the outer boundaries.
    outer_edge = (
        (jj == 0)
        | (jj == kmt.shape[0] - 1)
        | (ii == 0)
        | (ii == kmt.shape[1] - 1)
    )

    print("\nBoundary-location diagnostic")
    print("-" * 70)
    print(
        "Differing cells on outermost grid edge:",
        np.count_nonzero(outer_edge),
    )
    print(
        "Differing cells in grid interior:",
        np.count_nonzero(~outer_edge),
    )

    if np.all(outer_edge):
        print(
            "\nAll differences are on the outer edge. This may be due to "
            "halo/boundary treatment in one of the reference files."
        )


if __name__ == "__main__":
    main()
