import netCDF4
import numpy as np

filename = "cice_bc_from_NORESM.2023.nc"

with netCDF4.Dataset(filename) as nc:
    for name in [
        "aicen", "vicen", "vsnon",
        "alvln", "vlvln",
        "Tinz", "Sinz", "Tsnz", "Tsfc",
        "iage", "uvel", "vvel",
    ]:
        var = nc.variables[name]
        var.set_auto_mask(False)

        data = var[:]

        print(
            f"{name:8s} "
            f"NaN={np.count_nonzero(np.isnan(data)):10d}  "
            f"fill={np.count_nonzero(data == -9999.0):10d}  "
            f"min={np.nanmin(data):12.5g}  "
            f"max={np.nanmax(data):12.5g}"
        )
