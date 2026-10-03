import xarray as xr
import numpy as np

fname = "/cluster/shared/arcticfjord/NORESM/ice/NSSP585frc2_f09_tn14_20191105.cice.h.2023-01.nc"

with xr.open_dataset(fname) as ds:
    aice = ds["aice"].isel(time=0).values
    ardg = ds["ardg"].isel(time=0).values

    valid = (
        np.isfinite(aice)
        & np.isfinite(ardg)
        & (aice < 1e20)
        & (ardg < 1e20)
    )

    print("Maximum aice:", np.nanmax(aice[valid]))
    print("Maximum ardg:", np.nanmax(ardg[valid]))
    print("Maximum ardg - aice:", np.nanmax((ardg - aice)[valid]))
    print("Number of cells with ardg > aice:",
          np.count_nonzero((ardg > aice + 1e-6) & valid))
