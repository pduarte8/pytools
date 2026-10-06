#!/usr/bin/env python3

import netCDF4
import numpy as np

filename = "cice_bc_from_NORESM.trimmed.2025.nc"

boundary_names = [
    "aicen_W_bry",
    "aicen_E_bry",
    "aicen_S_bry",
    "aicen_N_bry",
]

with netCDF4.Dataset(filename, "r") as nc:

    for name in boundary_names:

        var = nc.variables[name]

        # Read stored numerical values directly; do not turn values into
        # masked arrays.
        var.set_auto_maskandscale(False)
        aicen = np.asarray(var[:], dtype=np.float64)

        # Dimensions: (TIME, ncat, boundary_point)
        aice_total = np.sum(aicen, axis=1)

        print("\n" + "=" * 70)
        print(name)
        print("shape:", aicen.shape)

        print("aicen minimum:", np.min(aicen))
        print("aicen maximum:", np.max(aicen))

        print("-9999 count:", np.count_nonzero(aicen == -9999.0))
        print("NaN count:", np.count_nonzero(np.isnan(aicen)))
        print("Inf count:", np.count_nonzero(np.isinf(aicen)))
        print("negative aicen count:", np.count_nonzero(aicen < 0.0))
        print("aicen > 1 count:", np.count_nonzero(aicen > 1.000001))

        print("sum(aicen) minimum:", np.min(aice_total))
        print("sum(aicen) maximum:", np.max(aice_total))
        print("negative sum(aicen) count:",
              np.count_nonzero(aice_total < 0.0))
        print("sum(aicen) > 1 count:",
              np.count_nonzero(aice_total > 1.000001))

        # Stop immediately if CICE could receive invalid ice area.
        if np.any(~np.isfinite(aicen)):
            raise ValueError(f"{name}: contains NaN or Inf.")

        if np.any(aicen < 0.0):
            raise ValueError(f"{name}: contains negative category area.")

        if np.any(aice_total < 0.0):
            raise ValueError(f"{name}: has negative total ice area.")

        if np.any(aice_total > 1.000001):
            raise ValueError(f"{name}: total ice area exceeds 1.")

print("\nPASS: all aicen boundary fields are finite and physically valid.")
