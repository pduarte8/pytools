import netCDF4
import numpy as np

filename = "cice_bc_from_NORESM.2023.nc"

with netCDF4.Dataset(filename) as nc:
    nc.set_auto_mask(False)

    aicen = nc.variables["aicen"][0]
    vicen = nc.variables["vicen"][0]
    vsnon = nc.variables["vsnon"][0]

valid = aicen[0] != -9999.0

aicen_clean = np.where(aicen == -9999.0, 0.0, aicen)
vicen_clean = np.where(vicen == -9999.0, 0.0, vicen)
vsnon_clean = np.where(vsnon == -9999.0, 0.0, vsnon)

aice_total = np.sum(aicen_clean, axis=0)
vice_total = np.sum(vicen_clean, axis=0)
vsno_total = np.sum(vsnon_clean, axis=0)

print("Minimum category concentration:", np.min(aicen_clean))
print("Maximum category concentration:", np.max(aicen_clean))
print("Maximum sum(aicen):", np.max(aice_total[valid]))

print("Minimum vicen:", np.min(vicen_clean))
print("Minimum vsnon:", np.min(vsnon_clean))

hice = np.divide(
    vice_total,
    aice_total,
    out=np.zeros_like(vice_total),
    where=aice_total > 1.0e-12,
)

hsnow = np.divide(
    vsno_total,
    aice_total,
    out=np.zeros_like(vsno_total),
    where=aice_total > 1.0e-12,
)

print("Maximum mean ice thickness over ice [m]:", np.max(hice))
print("Maximum mean snow depth over ice [m]:", np.max(hsnow))
