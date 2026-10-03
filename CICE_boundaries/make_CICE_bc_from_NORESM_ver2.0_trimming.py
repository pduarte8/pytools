#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create daily CICE boundary conditions from NORESM monthly CICE history output.

Confirmed NORESM category relationships:

    hi = sum(aicen * siitdthick, axis=category)
    hs = sum(aicen * siitdsnthick, axis=category)

Therefore:

    target aicen = NORESM aicen
    target vicen = NORESM aicen * NORESM siitdthick
    target vsnon = NORESM aicen * NORESM siitdsnthick

All category fields are constructed on the NORESM source grid before
bilinear regridding.

Daily fields are created by linear temporal interpolation between
monthly means placed at month midpoints. Previous December and next
January are included to interpolate all target-year days properly.
"""

import os
import sys
import subprocess
import tempfile
import datetime

import netCDF4
import numpy as np
import xarray as xr
from dateutil.relativedelta import relativedelta
from scipy.ndimage import distance_transform_edt


# =============================================================================
# USER SETTINGS
# =============================================================================

DIR_NORESM_ICE = "/cluster/shared/arcticfjord/NORESM/ice/"

ICE_FILE_TEMPLATE = (
    "NSSP585frc2_f09_tn14_20191105.cice.h.{date}.nc"
)

# Target CICE reference files.
REFGRID_T = "iceh.2015-11.selvar.tmask.nc"
REFGRID_U = "iceh.2015-11.selvar.sig2.nc"
REFGRID_ANGLE = "iceh.2015-11.selvar.ANGLE.nc"

# Dedicated target CICE mask file.
CICE_KMT_FILE = "/cluster/shared/arcticfjord/input_data/s800_cice_processed_a4_mod/cice.kmt.nc"

BC_FILE_PREFIX = "cice_bc_from_NORESM"

YEAR = 2024

# -------------------------------------------------------------------------
# TEST SETTINGS
#
# For a one-month test:
#
# TEST_START = "2023-01-01"
# TEST_END   = "2023-01-31"
#
# For the full year:
#
TEST_START = None
TEST_END   = None
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Target CICE configuration
# -------------------------------------------------------------------------

NCAT_TARGET = 5
NICE_LAYER = 7
NSNOW_LAYER = 1

# NORESM category maxima, as confirmed from the NCAT variable.
# The first four boundaries are compared within tolerance.
TARGET_CATEGORY_MAXIMA = np.array(
    [0.64, 1.39, 2.47, 4.57, 1000.0],
    dtype=np.float64,
)

CATEGORY_BOUNDARY_TOLERANCE = 0.01  # m

# -------------------------------------------------------------------------
# Numerical / physical constants
# -------------------------------------------------------------------------

FILL_VALUE = -9999.0
SOURCE_FILL_LIMIT = 1.0e20

EPS_AICE = 1.0e-12
SNOW_EPS = 1.0e-6  # m

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY

# Safety threshold for remapped velocities [m s-1].
MAX_ICE_SPEED = 10.0

# Bilinear remapping is used because the target regional grid lacks the
# outer U-grid halo needed to construct complete target-cell polygons
# for remapcon.
REMAP_METHOD = "remapbil"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_cdo(command):
    """Run a CDO command and stop on failure."""
    print("\nRunning:")
    print(command)

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("\nCDO stdout:")
        print(result.stdout)

        print("\nCDO stderr:")
        print(result.stderr)

        raise RuntimeError("CDO command failed.")

    return result


def remove_if_exists(filename):
    """Remove file if it exists."""
    if os.path.exists(filename):
        os.remove(filename)


def clean_array(data):
    """
    Convert masked/fill values to NaN and return float64 NumPy array.
    """
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)

    output = np.asarray(data, dtype=np.float64).copy()

    output[np.abs(output) > SOURCE_FILL_LIMIT] = np.nan

    return output


def replace_invalid_with_fill(data):
    """
    Replace NaN, Inf, and extreme values by FILL_VALUE before writing.
    """
    output = np.asarray(data, dtype=np.float64).copy()

    invalid = (
        ~np.isfinite(output)
        | (np.abs(output) > SOURCE_FILL_LIMIT)
    )

    output[invalid] = FILL_VALUE

    return output


def month_midpoint(year, month):
    """
    Return the midpoint of a calendar month, rounded to midnight.

    Monthly means are represented by this time during interpolation.
    """
    month_start = datetime.datetime(year, month, 1)
    next_month = month_start + relativedelta(months=1)

    midpoint = month_start + (next_month - month_start) / 2

    rounded_day = (
        midpoint + datetime.timedelta(hours=12)
    ).date()

    return datetime.datetime.combine(
        rounded_day,
        datetime.time(0, 0),
    )


def monthly_date_strings(year):
    """
    Return:
        previous December,
        January ... December target year,
        following January.
    """
    dates = [f"{year - 1}-12"]

    dates.extend(
        f"{year}-{month:02d}"
        for month in range(1, 13)
    )

    dates.append(f"{year + 1}-01")

    return dates


def create_cdo_grid_descriptor(
    ncfile,
    gridfile,
    lon_name,
    lat_name,
):
    """
    Create a CDO curvilinear-grid descriptor using grid-cell centres.

    This is sufficient for remapbil.
    """
    with xr.open_dataset(ncfile) as ds:
        lon = clean_array(ds[lon_name].values)
        lat = clean_array(ds[lat_name].values)

    # Use [-180, 180] longitude convention consistently.
    if np.nanmax(lon) > 180.0:
        lon = ((lon + 180.0) % 360.0) - 180.0

    lon = np.nan_to_num(lon, nan=1.0e20)
    lat = np.nan_to_num(lat, nan=1.0e20)

    nj, ni = lat.shape

    with open(gridfile, "w") as f:
        f.write("gridtype = curvilinear\n")
        f.write(f"gridsize = {nj * ni}\n")
        f.write(f"xsize = {ni}\n")
        f.write(f"ysize = {nj}\n")

        f.write(
            "xvals = "
            + " ".join(map(str, lon.flatten(order="C")))
            + "\n"
        )

        f.write(
            "yvals = "
            + " ".join(map(str, lat.flatten(order="C")))
            + "\n"
        )

    print(f"Created CDO grid descriptor: {gridfile}")


def remap_file(
    infile,
    source_grid,
    target_grid,
    outfile,
    workdir,
):
    """
    Apply source grid and bilinearly remap to target grid.
    """
    basename = os.path.basename(infile).replace(".nc", "")

    infile_with_grid = os.path.join(
        workdir,
        basename + "_with_grid.nc",
    )

    run_cdo(
        f"cdo -O setgrid,{source_grid} "
        f"{infile} {infile_with_grid}"
    )

    run_cdo(
        f"cdo -O {REMAP_METHOD},{target_grid} "
        f"{infile_with_grid} {outfile}"
    )


def remap_dataarray(
    da,
    source_grid,
    target_grid,
    workdir,
    suffix,
):
    """
    Regrid one DataArray and return a NumPy array.

    The DataArray must have a name.
    """
    if da.name is None:
        raise ValueError("Input DataArray has no variable name.")

    infile = os.path.join(
        workdir,
        f"{da.name}_{suffix}_input.nc",
    )

    outfile = os.path.join(
        workdir,
        f"{da.name}_{suffix}_remapped.nc",
    )

    da.to_dataset(name=da.name).to_netcdf(infile)

    remap_file(
        infile=infile,
        source_grid=source_grid,
        target_grid=target_grid,
        outfile=outfile,
        workdir=workdir,
    )

    with xr.open_dataset(outfile) as ds_out:
        output = clean_array(
            ds_out[da.name].load().values
        )

    return output


def remap_category_field(
    da,
    source_grid,
    target_grid,
    workdir,
):
    """
    Regrid category-dependent variable one category at a time.

    Input:
        (time, nc, nj, ni)

    Output:
        (time, nc, target_nj, target_ni)
    """
    if "nc" not in da.dims:
        raise ValueError(
            f"Variable {da.name} has no category dimension 'nc'."
        )

    ncat_source = da.sizes["nc"]

    category_arrays = []

    for nc_idx in range(ncat_source):
        print(
            f"  Remapping {da.name}, "
            f"category {nc_idx + 1}/{ncat_source}"
        )

        da_cat = da.isel(nc=nc_idx).copy()
        da_cat.name = da.name

        remapped = remap_dataarray(
            da=da_cat,
            source_grid=source_grid,
            target_grid=target_grid,
            workdir=workdir,
            suffix=f"cat_{nc_idx}",
        )

        if remapped.ndim != 3:
            raise ValueError(
                f"Unexpected dimensions for remapped {da.name}: "
                f"{remapped.shape}; expected (time, nj, ni)."
            )

        category_arrays.append(remapped)

    return np.stack(category_arrays, axis=1)


def nearest_fill_2d(
    field,
    ocean_mask,
):
    """
    Fill NaNs at valid ocean cells using nearest valid ocean-cell value.

    Target land remains NaN.

    This is used only when remapping leaves coastal/ocean gaps. It
    prevents FILL_VALUE from entering valid target ocean cells.
    """
    field = np.asarray(field, dtype=np.float64).copy()

    valid = (
        ocean_mask
        & np.isfinite(field)
    )

    missing = (
        ocean_mask
        & ~np.isfinite(field)
    )

    if not np.any(missing):
        field[~ocean_mask] = np.nan
        return field

    if not np.any(valid):
        raise ValueError(
            "Cannot nearest-fill field: no valid ocean values exist."
        )

    _, indices = distance_transform_edt(
        ~valid,
        return_indices=True,
    )

    field[missing] = field[
        indices[0][missing],
        indices[1][missing],
    ]

    field[~ocean_mask] = np.nan

    return field


def nearest_fill_category_field(
    field,
    ocean_mask,
):
    """
    Apply nearest_fill_2d independently to every category.
    """
    field = np.asarray(field, dtype=np.float64).copy()

    for nc_idx in range(field.shape[0]):
        field[nc_idx] = nearest_fill_2d(
            field[nc_idx],
            ocean_mask,
        )

    return field


def sanitize_monthly_category_state(
    aicen,
    vicen,
    vsnon,
    ocean_mask,
):
    """
    Clean one monthly category state after remapping.

    All valid target ocean cells are filled if remapping left gaps.
    Land cells remain NaN internally and become FILL_VALUE on output.
    """
    aicen = nearest_fill_category_field(
        aicen,
        ocean_mask,
    )

    vicen = nearest_fill_category_field(
        vicen,
        ocean_mask,
    )

    vsnon = nearest_fill_category_field(
        vsnon,
        ocean_mask,
    )

    # Clip numerical interpolation artefacts.
    aicen = np.maximum(aicen, 0.0)
    vicen = np.maximum(vicen, 0.0)
    vsnon = np.maximum(vsnon, 0.0)

    total_aice = np.nansum(aicen, axis=0)

    if np.nanmax(total_aice[ocean_mask]) > 1.001:
        raise ValueError(
            "Monthly sum(aicen) exceeds 1.001 after regridding. "
            "Check the source/target grids."
        )

    # If bilinear interpolation makes concentration slightly exceed one,
    # scale all related category state fields together.
    excess = (
        ocean_mask
        & (total_aice > 1.0)
    )

    if np.any(excess):
        print(
            "WARNING: monthly sum(aicen) > 1 in "
            f"{np.count_nonzero(excess)} cells; "
            "scaling category area/volume/snow consistently."
        )

        factor = np.ones_like(total_aice)
        factor[excess] = 1.0 / total_aice[excess]

        aicen *= factor[None, :, :]
        vicen *= factor[None, :, :]
        vsnon *= factor[None, :, :]

    return aicen, vicen, vsnon


def interpolate_linear(
    field_left,
    field_right,
    fraction,
):
    """
    Linearly interpolate two fields.

    fraction = 0 means left record;
    fraction = 1 means right record.
    """
    return (
        (1.0 - fraction) * field_left
        + fraction * field_right
    )


# =============================================================================
# MONTHLY PROCESSING
# =============================================================================

def process_month(
    date_str,
    source_t_grid,
    source_u_grid,
    target_t_grid,
    target_u_grid,
    target_angle,
    ocean_mask,
    workdir,
):
    """
    Process one NORESM monthly file and return target-grid monthly fields.

    The returned dictionary contains only fields that should be temporally
    interpolated. Derived thermodynamic and level-ice fields are calculated
    after daily interpolation.
    """
    infile = os.path.join(
        DIR_NORESM_ICE,
        ICE_FILE_TEMPLATE.format(date=date_str),
    )

    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Missing NORESM CICE file:\n{infile}"
        )

    print("\n" + "=" * 80)
    print(f"Processing source month: {date_str}")
    print("=" * 80)

    with xr.open_dataset(
        infile,
        mask_and_scale=True,
    ) as ds:

        required = [
            "aicen",
            "siitdthick",
            "siitdsnthick",
            "hi",
            "hs",
            "siage",
            "ardg",
            "sitemptop",
            "sitempsnic",
            "sitempbot",
            "siu",
            "siv",
            "ANGLE",
            "NCAT",
        ]

        missing = [
            name for name in required
            if name not in ds.variables
        ]

        if missing:
            raise KeyError(
                "Required NORESM variables missing:\n"
                + "\n".join(missing)
            )

        # -----------------------------------------------------------------
        # Category consistency checks.
        # -----------------------------------------------------------------

        if ds.sizes["nc"] != NCAT_TARGET:
            raise ValueError(
                "Different category counts:\n"
                f"NORESM = {ds.sizes['nc']}\n"
                f"Target = {NCAT_TARGET}"
            )

        source_category_maxima = clean_array(
            ds["NCAT"].values
        )

        if not np.allclose(
            source_category_maxima[:-1],
            TARGET_CATEGORY_MAXIMA[:-1],
            rtol=0.0,
            atol=CATEGORY_BOUNDARY_TOLERANCE,
        ):
            raise ValueError(
                "NORESM and target finite category maxima differ:\n"
                f"NORESM: {source_category_maxima}\n"
                f"Target: {TARGET_CATEGORY_MAXIMA}"
            )

        # Final category is open-ended. 1000 m and 1e8 m both mean
        # practically unlimited maximum thickness.
        if source_category_maxima[-1] < 100.0:
            raise ValueError(
                "Final NORESM category is not effectively open-ended:\n"
                f"{source_category_maxima[-1]}"
            )

        # -----------------------------------------------------------------
        # Construct target-CICE-equivalent category state on source grid.
        # -----------------------------------------------------------------

        aicen_source = ds["aicen"].copy()
        aicen_source.name = "aicen"

        # Confirmed:
        # hi = sum(aicen * siitdthick)
        vicen_source = (
            ds["aicen"]
            * ds["siitdthick"]
        )
        vicen_source.name = "vicen"

        # Confirmed:
        # hs = sum(aicen * siitdsnthick)
        vsnon_source = (
            ds["aicen"]
            * ds["siitdsnthick"]
        )
        vsnon_source.name = "vsnon"

        # Source consistency diagnostics.
        hi_error = clean_array(
            (
                vicen_source.sum(dim="nc")
                - ds["hi"]
            ).isel(time=0).values
        )

        hs_error = clean_array(
            (
                vsnon_source.sum(dim="nc")
                - ds["hs"]
            ).isel(time=0).values
        )

        print(
            "Maximum source hi consistency error [m]:",
            np.nanmax(np.abs(hi_error)),
        )

        print(
            "Maximum source hs consistency error [m]:",
            np.nanmax(np.abs(hs_error)),
        )

        # -----------------------------------------------------------------
        # Category state remapping.
        # -----------------------------------------------------------------

        aicen_all = remap_category_field(
            da=aicen_source,
            source_grid=source_t_grid,
            target_grid=target_t_grid,
            workdir=workdir,
        )

        vicen_all = remap_category_field(
            da=vicen_source,
            source_grid=source_t_grid,
            target_grid=target_t_grid,
            workdir=workdir,
        )

        vsnon_all = remap_category_field(
            da=vsnon_source,
            source_grid=source_t_grid,
            target_grid=target_t_grid,
            workdir=workdir,
        )

        aicen = aicen_all[0]
        vicen = vicen_all[0]
        vsnon = vsnon_all[0]

        aicen, vicen, vsnon = sanitize_monthly_category_state(
            aicen=aicen,
            vicen=vicen,
            vsnon=vsnon,
            ocean_mask=ocean_mask,
        )

        # -----------------------------------------------------------------
        # Remap scalar diagnostic fields on CICE T grid.
        # -----------------------------------------------------------------

        scalar_names = [
            "siage",
            "ardg",
            "sitemptop",
            "sitempsnic",
            "sitempbot",
        ]

        scalars = {}

        for varname in scalar_names:
            da = ds[varname].copy()
            da.name = varname

            remapped = remap_dataarray(
                da=da,
                source_grid=source_t_grid,
                target_grid=target_t_grid,
                workdir=workdir,
                suffix="scalar",
            )

            if remapped.ndim != 3:
                raise ValueError(
                    f"Unexpected dimensions for {varname}: "
                    f"{remapped.shape}"
                )

            field = remapped[0]

            # Fill valid ocean remapping gaps from nearest valid target
            # ocean cell. Keep target land as NaN internally.
            field = nearest_fill_2d(
                field,
                ocean_mask,
            )

            scalars[varname] = field

        # Convert CICE temperature diagnostics from K to deg C.
        sitemptop = scalars["sitemptop"] - 273.15
        sitempsnic = scalars["sitempsnic"] - 273.15
        sitempbot = scalars["sitempbot"] - 273.15

        # siage is in seconds.
        siage_seconds = np.maximum(
            scalars["siage"],
            0.0,
        )

        # ardg is a total-grid-cell ridged area fraction.
        # It must be between zero and total ice area.
        aice_total = np.sum(aicen, axis=0)

        ardg = np.clip(
            scalars["ardg"],
            0.0,
            aice_total,
        )

        # -----------------------------------------------------------------
        # Velocity:
        #
        # source grid-relative components
        # -> earth-relative east/north
        # -> remapbil
        # -> target-grid components.
        # -----------------------------------------------------------------

        source_angle = ds["ANGLE"]

        if source_angle.shape != ds["siu"].shape[-2:]:
            raise ValueError(
                "NORESM ANGLE and siu dimensions differ:\n"
                f"ANGLE = {source_angle.shape}\n"
                f"siu   = {ds['siu'].shape}"
            )

        u_east = (
            ds["siu"] * np.cos(source_angle)
            - ds["siv"] * np.sin(source_angle)
        )
        u_east.name = "u_east"

        v_north = (
            ds["siu"] * np.sin(source_angle)
            + ds["siv"] * np.cos(source_angle)
        )
        v_north.name = "v_north"

        velocity_input = os.path.join(
            workdir,
            "velocity_earth_relative_input.nc",
        )

        velocity_output = os.path.join(
            workdir,
            "velocity_earth_relative_remapped.nc",
        )

        xr.Dataset(
            {
                "u_east": u_east,
                "v_north": v_north,
            }
        ).to_netcdf(velocity_input)

        remap_file(
            infile=velocity_input,
            source_grid=source_u_grid,
            target_grid=target_u_grid,
            outfile=velocity_output,
            workdir=workdir,
        )

        with xr.open_dataset(velocity_output) as ds_velocity:
            u_east_target = clean_array(
                ds_velocity["u_east"]
                .isel(time=0)
                .load()
                .values
            )

            v_north_target = clean_array(
                ds_velocity["v_north"]
                .isel(time=0)
                .load()
                .values
            )

        # Fill remapping gaps over target ocean.
        u_east_target = nearest_fill_2d(
            u_east_target,
            ocean_mask,
        )

        v_north_target = nearest_fill_2d(
            v_north_target,
            ocean_mask,
        )

        # Earth-relative -> target CICE grid axes.
        uvel = (
            u_east_target * np.cos(target_angle)
            + v_north_target * np.sin(target_angle)
        )

        vvel = (
            -u_east_target * np.sin(target_angle)
            + v_north_target * np.cos(target_angle)
        )

        bad_velocity = (
            ~np.isfinite(uvel)
            | ~np.isfinite(vvel)
            | (np.abs(uvel) > MAX_ICE_SPEED)
            | (np.abs(vvel) > MAX_ICE_SPEED)
        )

        bad_velocity_ocean = (
            ocean_mask
            & bad_velocity
        )

        if np.any(bad_velocity_ocean):
            print(
                "WARNING: replacing "
                f"{np.count_nonzero(bad_velocity_ocean)} "
                "invalid/unrealistic target ocean velocity values with zero."
            )

            uvel[bad_velocity_ocean] = 0.0
            vvel[bad_velocity_ocean] = 0.0

        uvel[~ocean_mask] = np.nan
        vvel[~ocean_mask] = np.nan

    return {
        "aicen": aicen,
        "vicen": vicen,
        "vsnon": vsnon,
        "siage_seconds": siage_seconds,
        "ardg": ardg,
        "sitemptop": sitemptop,
        "sitempsnic": sitempsnic,
        "sitempbot": sitempbot,
        "uvel": uvel,
        "vvel": vvel,
    }


# =============================================================================
# DAILY DERIVED FIELDS
# =============================================================================

def derive_daily_fields(
    state,
    ocean_mask,
):
    """
    Apply physical bounds and derive CICE boundary fields for one day.
    """
    aicen = np.asarray(state["aicen"], dtype=np.float64).copy()
    vicen = np.asarray(state["vicen"], dtype=np.float64).copy()
    vsnon = np.asarray(state["vsnon"], dtype=np.float64).copy()

    # Interpolated fields can have small negative values.
    aicen = np.maximum(aicen, 0.0)
    vicen = np.maximum(vicen, 0.0)
    vsnon = np.maximum(vsnon, 0.0)

    # Keep target land as NaN internally.
    aicen[:, ~ocean_mask] = np.nan
    vicen[:, ~ocean_mask] = np.nan
    vsnon[:, ~ocean_mask] = np.nan

    aice_total = np.nansum(aicen, axis=0)

    # If temporal interpolation yields sum(aicen) > 1, scale all
    # category state fields consistently.
    excess = (
        ocean_mask
        & (aice_total > 1.0)
    )

    if np.any(excess):
        factor = np.ones_like(aice_total)
        factor[excess] = 1.0 / aice_total[excess]

        aicen *= factor[None, :, :]
        vicen *= factor[None, :, :]
        vsnon *= factor[None, :, :]

        aice_total = np.nansum(aicen, axis=0)

    # Clip category values individually to physically valid ranges.
    aicen = np.minimum(aicen, 1.0)

    # -----------------------------------------------------------------
    # Ridged and level ice.
    #
    # ardg is a ridged ice area fraction over the total grid-cell area:
    #
    # 0 <= ardg <= total aice.
    #
    # Since ardg has no category dimension, distribute level/ridged
    # partition proportionally over categories.
    # -----------------------------------------------------------------

    ardg = np.asarray(state["ardg"], dtype=np.float64).copy()
    ardg = np.maximum(ardg, 0.0)
    ardg = np.minimum(ardg, aice_total)
    ardg[~ocean_mask] = np.nan

    level_factor = np.divide(
        aice_total - ardg,
        aice_total,
        out=np.zeros_like(aice_total),
        where=aice_total > EPS_AICE,
    )

    level_factor = np.clip(
        level_factor,
        0.0,
        1.0,
    )

    alvln = aicen * level_factor[None, :, :]
    vlvln = vicen * level_factor[None, :, :]

    # -----------------------------------------------------------------
    # Age.
    # -----------------------------------------------------------------

    siage_seconds = np.maximum(
        np.asarray(
            state["siage_seconds"],
            dtype=np.float64,
        ),
        0.0,
    )

    siage_seconds[~ocean_mask] = np.nan

    iage_days = siage_seconds / SECONDS_PER_DAY
    siage_years = siage_seconds / SECONDS_PER_YEAR

    # -----------------------------------------------------------------
    # Thermodynamics.
    #
    # sitemptop:
    #     snow surface temperature if snow exists;
    #     ice-top temperature if snow does not exist.
    #
    # sitempsnic:
    #     snow–ice interface temperature.
    #
    # sitempbot:
    #     ice-bottom temperature.
    # -----------------------------------------------------------------

    sitemptop = np.asarray(
        state["sitemptop"],
        dtype=np.float64,
    )

    sitempsnic = np.asarray(
        state["sitempsnic"],
        dtype=np.float64,
    )

    sitempbot = np.asarray(
        state["sitempbot"],
        dtype=np.float64,
    )

    sitemptop[~ocean_mask] = np.nan
    sitempsnic[~ocean_mask] = np.nan
    sitempbot[~ocean_mask] = np.nan

    valid_ice = (
        aicen > EPS_AICE
    )

    # Category-local ice thickness and snow depth.
    hice_category = np.divide(
        vicen,
        aicen,
        out=np.zeros_like(vicen),
        where=valid_ice,
    )

    hsnow_category = np.divide(
        vsnon,
        aicen,
        out=np.zeros_like(vsnon),
        where=valid_ice,
    )

    hice_category = np.maximum(
        hice_category,
        0.0,
    )

    hsnow_category = np.maximum(
        hsnow_category,
        0.0,
    )

    snow_present = (
        hsnow_category > SNOW_EPS
    )

    ncat, nj, ni = aicen.shape

    # Snow/ice surface temperature in each category.
    surface_temp = np.broadcast_to(
        sitemptop,
        (ncat, nj, ni),
    )

    snow_ice_interface_temp = np.broadcast_to(
        sitempsnic,
        (ncat, nj, ni),
    )

    bottom_temp = np.broadcast_to(
        sitempbot,
        (ncat, nj, ni),
    )

    # Ice top:
    # - snow-covered category -> snow–ice interface;
    # - snow-free category -> exposed ice surface.
    ice_top_temp = np.where(
        snow_present,
        snow_ice_interface_temp,
        surface_temp,
    )

    # One snow layer.
    tsnow = np.zeros(
        (ncat, NSNOW_LAYER, nj, ni),
        dtype=np.float64,
    )

    if NSNOW_LAYER != 1:
        raise NotImplementedError(
            "This script currently assumes NSNOW_LAYER == 1."
        )

    tsnow[:, 0, :, :] = np.where(
        snow_present,
        0.5 * (
            surface_temp
            + snow_ice_interface_temp
        ),
        0.0,
    )

    # Ice-layer temperatures.
    t_ice = np.zeros(
        (ncat, NICE_LAYER, nj, ni),
        dtype=np.float64,
    )

    for k in range(NICE_LAYER):
        z = (k + 0.5) / float(NICE_LAYER)

        t_ice[:, k, :, :] = (
            ice_top_temp
            + z * (
                bottom_temp
                - ice_top_temp
            )
        )

    # Ice salinity profiles.
    #
    # FYI: siage <= 1 year
    # MYI: siage > 1 year
    is_myi = np.broadcast_to(
        siage_years > 1.0,
        (ncat, nj, ni),
    )

    s_ice = np.zeros(
        (ncat, NICE_LAYER, nj, ni),
        dtype=np.float64,
    )

    for k in range(NICE_LAYER):
        z = (k + 0.5) / float(NICE_LAYER)

        # MYI profile.
        a_myi = 0.407
        b_myi = 0.573
        smax_myi = 3.2

        theta = np.pi * z ** (
            a_myi / (z + b_myi)
        )

        salinity_myi = (
            0.5
            * smax_myi
            * (1.0 - np.cos(theta))
        )

        # FYI profile.
        salinity_fyi = (
            19.539 * z**2
            - 19.93 * z
            + 8.913
        )

        s_ice[:, k, :, :] = np.where(
            is_myi,
            salinity_myi,
            salinity_fyi,
        )

    # Zero thermodynamic values in open-water categories.
    t_ice = np.where(
        valid_ice[:, None, :, :],
        t_ice,
        0.0,
    )

    s_ice = np.where(
        valid_ice[:, None, :, :],
        s_ice,
        0.0,
    )

    tsnow = np.where(
        valid_ice[:, None, :, :],
        tsnow,
        0.0,
    )

    tsfc = np.where(
        valid_ice,
        surface_temp,
        0.0,
    )

    # -----------------------------------------------------------------
    # Velocity.
    # -----------------------------------------------------------------

    uvel = np.asarray(
        state["uvel"],
        dtype=np.float64,
    ).copy()

    vvel = np.asarray(
        state["vvel"],
        dtype=np.float64,
    ).copy()

    bad_velocity = (
        ~np.isfinite(uvel)
        | ~np.isfinite(vvel)
        | (np.abs(uvel) > MAX_ICE_SPEED)
        | (np.abs(vvel) > MAX_ICE_SPEED)
    )

    bad_velocity_ocean = (
        ocean_mask
        & bad_velocity
    )

    if np.any(bad_velocity_ocean):
        print(
            "WARNING: daily velocity has "
            f"{np.count_nonzero(bad_velocity_ocean)} "
            "invalid/unrealistic ocean values; replacing with zero."
        )

        uvel[bad_velocity_ocean] = 0.0
        vvel[bad_velocity_ocean] = 0.0

    uvel[~ocean_mask] = np.nan
    vvel[~ocean_mask] = np.nan

    # Pond and brine fields are not derived from NORESM here.
    shape_cat = aicen.shape

    apondn = np.zeros(shape_cat, dtype=np.float64)
    hpondn = np.zeros(shape_cat, dtype=np.float64)
    ipondn = np.zeros(shape_cat, dtype=np.float64)
    fbrine = np.zeros(shape_cat, dtype=np.float64)
    hbrine = np.zeros(shape_cat, dtype=np.float64)

    # Keep land internally NaN.
    for field in [
        apondn,
        hpondn,
        ipondn,
        fbrine,
        hbrine,
    ]:
        field[:, ~ocean_mask] = np.nan

    return {
        "aicen": aicen,
        "vicen": vicen,
        "vsnon": vsnon,
        "alvln": alvln,
        "vlvln": vlvln,
        "Tinz": t_ice,
        "Sinz": s_ice,
        "Tsnz": tsnow,
        "Tsfc": tsfc,
        "iage": iage_days,
        "apondn": apondn,
        "hpondn": hpondn,
        "ipondn": ipondn,
        "fbrine": fbrine,
        "hbrine": hbrine,
        "uvel": uvel,
        "vvel": vvel,
        "aice_total": aice_total,
        "ardg": ardg,
    }


# =============================================================================
# OUTPUT FILE FUNCTIONS
# =============================================================================

def create_output_variable(
    nc_out,
    name,
    dimensions,
    units,
    long_name,
):
    """
    Create output variable with both _FillValue and missing_value.
    """
    variable = nc_out.createVariable(
        name,
        "f8",
        dimensions,
        fill_value=FILL_VALUE,
        zlib=False,
    )

    variable.missing_value = FILL_VALUE
    variable.units = units
    variable.long_name = long_name

    return variable


def create_boundary_file(
    outfile,
    target_tlat,
    target_tlon,
    target_ulat,
    target_ulon,
):
    """
    Create full-domain daily boundary file.
    """
    remove_if_exists(outfile)

    nj, ni = target_tlat.shape

    nc_out = netCDF4.Dataset(
        outfile,
        "w",
        format="NETCDF4",
    )

    nc_out.createDimension("TIME", None)
    nc_out.createDimension("eta_t", nj)
    nc_out.createDimension("xi_t", ni)
    nc_out.createDimension("ncat", NCAT_TARGET)
    nc_out.createDimension("nkice", NICE_LAYER)
    nc_out.createDimension("nksnow", NSNOW_LAYER)

    # Index variables.
    var = nc_out.createVariable(
        "Domain_columns",
        "i2",
        ("eta_t",),
    )
    var[:] = np.arange(nj)

    var = nc_out.createVariable(
        "Domain_lines",
        "i2",
        ("xi_t",),
    )
    var[:] = np.arange(ni)

    var = nc_out.createVariable(
        "NCAT",
        "i2",
        ("ncat",),
    )
    var[:] = np.arange(NCAT_TARGET)

    var = nc_out.createVariable(
        "Ice_layers",
        "i2",
        ("nkice",),
    )
    var[:] = np.arange(NICE_LAYER)

    var = nc_out.createVariable(
        "Snow_layers",
        "i2",
        ("nksnow",),
    )
    var[:] = np.arange(NSNOW_LAYER)

    # Geographic coordinates.
    TLAT = nc_out.createVariable(
        "TLAT",
        "f4",
        ("eta_t", "xi_t"),
    )
    TLAT[:, :] = target_tlat
    TLAT.units = "degrees_north"

    TLON = nc_out.createVariable(
        "TLON",
        "f4",
        ("eta_t", "xi_t"),
    )
    TLON[:, :] = target_tlon
    TLON.units = "degrees_east"

    ULAT = nc_out.createVariable(
        "ULAT",
        "f4",
        ("eta_t", "xi_t"),
    )
    ULAT[:, :] = target_ulat
    ULAT.units = "degrees_north"

    ULON = nc_out.createVariable(
        "ULON",
        "f4",
        ("eta_t", "xi_t"),
    )
    ULON[:, :] = target_ulon
    ULON.units = "degrees_east"

    # Time.
    time = nc_out.createVariable(
        "time",
        "f8",
        ("TIME",),
    )
    time.standard_name = "time"
    time.long_name = "time"
    time.units = "days since 1900-01-01 00:00:00"
    time.calendar = "gregorian"
    time.axis = "T"

    nc_out.createVariable(
        "Time",
        "i2",
        ("TIME",),
    )

    # Category ice state.
    create_output_variable(
        nc_out,
        "aicen",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "1",
        "category ice area fraction",
    )

    create_output_variable(
        nc_out,
        "vicen",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "m",
        "category ice volume per total grid-cell area",
    )

    create_output_variable(
        nc_out,
        "vsnon",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "m",
        "category snow volume per total grid-cell area",
    )

    # Level ice.
    create_output_variable(
        nc_out,
        "alvln",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "1",
        "category level ice area fraction",
    )

    create_output_variable(
        nc_out,
        "vlvln",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "m",
        "category level ice volume per total grid-cell area",
    )

    # Thermodynamics.
    create_output_variable(
        nc_out,
        "Tinz",
        ("TIME", "ncat", "nkice", "eta_t", "xi_t"),
        "deg_C",
        "ice layer temperature",
    )

    create_output_variable(
        nc_out,
        "Sinz",
        ("TIME", "ncat", "nkice", "eta_t", "xi_t"),
        "psu",
        "ice layer salinity",
    )

    create_output_variable(
        nc_out,
        "Tsnz",
        ("TIME", "ncat", "nksnow", "eta_t", "xi_t"),
        "deg_C",
        "snow layer temperature",
    )

    create_output_variable(
        nc_out,
        "Tsfc",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "deg_C",
        "ice or snow surface temperature",
    )

    # Age.
    create_output_variable(
        nc_out,
        "iage",
        ("TIME", "eta_t", "xi_t"),
        "day",
        "sea ice age",
    )

    # Ponds and brine.
    create_output_variable(
        nc_out,
        "apondn",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "1",
        "melt pond fraction",
    )

    create_output_variable(
        nc_out,
        "hpondn",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "m",
        "melt pond depth",
    )

    create_output_variable(
        nc_out,
        "ipondn",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "m",
        "pond ice thickness",
    )

    create_output_variable(
        nc_out,
        "fbrine",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "1",
        "brine fraction",
    )

    create_output_variable(
        nc_out,
        "hbrine",
        ("TIME", "ncat", "eta_t", "xi_t"),
        "m",
        "brine height above ice base",
    )

    # Velocity.
    create_output_variable(
        nc_out,
        "uvel",
        ("TIME", "eta_t", "xi_t"),
        "m s-1",
        "ice velocity x component in target CICE grid axes",
    )

    create_output_variable(
        nc_out,
        "vvel",
        ("TIME", "eta_t", "xi_t"),
        "m s-1",
        "ice velocity y component in target CICE grid axes",
    )

    return nc_out


def apply_land_fill(
    field,
    ocean_mask,
):
    """
    Apply fill value to target land cells.

    Supports 2-D, 3-D and 4-D fields whose final dimensions are nj, ni.
    """
    field = np.asarray(field, dtype=np.float64)

    if field.ndim == 2:
        output = np.where(
            ocean_mask,
            field,
            FILL_VALUE,
        )

    elif field.ndim == 3:
        output = np.where(
            ocean_mask[None, :, :],
            field,
            FILL_VALUE,
        )

    elif field.ndim == 4:
        output = np.where(
            ocean_mask[None, None, :, :],
            field,
            FILL_VALUE,
        )

    else:
        raise ValueError(
            f"Unsupported field dimensions: {field.shape}"
        )

    return replace_invalid_with_fill(output)


def make_trimmed_boundary_file(
    full_file,
    trimmed_file,
):
    """
    Create W/E/S/N boundary fields from the full-domain file.
    """
    remove_if_exists(trimmed_file)

    with netCDF4.Dataset(full_file, "r") as src, netCDF4.Dataset(
        trimmed_file,
        "w",
        format="NETCDF3_64BIT_OFFSET",
    ) as dst:

        # Dimensions.
        for dim_name, dim in src.dimensions.items():
            dst.createDimension(
                dim_name,
                None if dim.isunlimited() else len(dim),
            )

        for name, src_var in src.variables.items():

            dims = src_var.dimensions

            # Coordinates and index fields are copied directly.
            if len(dims) < 3:
                dst_var = dst.createVariable(
                    name,
                    src_var.datatype,
                    dims,
                )

                attributes = {
                    key: value
                    for key, value in src_var.__dict__.items()
                    if key != "_FillValue"
                }

                dst_var.setncatts(attributes)
                dst_var[:] = src_var[:]

                continue

            fill = getattr(
                src_var,
                "_FillValue",
                FILL_VALUE,
            )

            attributes = {
                key: value
                for key, value in src_var.__dict__.items()
                if key != "_FillValue"
            }

            # West and east boundaries.
            for suffix, x_index in [
                ("_W_bry", 0),
                ("_E_bry", -1),
            ]:
                out_dims = dims[:-1]

                dst_var = dst.createVariable(
                    name + suffix,
                    src_var.datatype,
                    out_dims,
                    fill_value=fill,
                )

                dst_var.setncatts(attributes)

                indexer = [slice(None)] * (len(dims) - 1)
                indexer.append(x_index)

                dst_var[:] = src_var[tuple(indexer)]

            # South and north boundaries.
            for suffix, y_index in [
                ("_S_bry", 0),
                ("_N_bry", -1),
            ]:
                out_dims = dims[:-2] + dims[-1:]

                dst_var = dst.createVariable(
                    name + suffix,
                    src_var.datatype,
                    out_dims,
                    fill_value=fill,
                )

                dst_var.setncatts(attributes)

                indexer = [slice(None)] * (len(dims) - 2)
                indexer.append(y_index)
                indexer.append(slice(None))

                dst_var[:] = src_var[tuple(indexer)]


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Validate files.
    # -------------------------------------------------------------------------

    for filename in [
        REFGRID_T,
        REFGRID_U,
        REFGRID_ANGLE,
        CICE_KMT_FILE,
    ]:
        if not os.path.exists(filename):
            sys.exit(
                f"Required target file not found:\n{filename}"
            )

    source_months = monthly_date_strings(YEAR)

    print("Monthly source files required:")
    print(source_months)

    for date_str in source_months:
        filename = os.path.join(
            DIR_NORESM_ICE,
            ICE_FILE_TEMPLATE.format(date=date_str),
        )

        if not os.path.exists(filename):
            sys.exit(
                "Required source month is missing:\n"
                f"{filename}\n\n"
                "Daily interpolation requires previous December and "
                "following January as well as the target-year files."
            )

    first_ice_file = os.path.join(
        DIR_NORESM_ICE,
        ICE_FILE_TEMPLATE.format(date=source_months[0]),
    )

    # -------------------------------------------------------------------------
    # Target mask.
    # -------------------------------------------------------------------------

    with netCDF4.Dataset(CICE_KMT_FILE, "r") as nc_kmt:
        target_ocean_mask = clean_array(
            nc_kmt.variables["kmt"][:]
        ) > 0

    # -------------------------------------------------------------------------
    # Create CDO grid descriptors.
    # -------------------------------------------------------------------------

    source_t_grid = "source_t_grid.txt"
    source_u_grid = "source_u_grid.txt"

    target_t_grid = "target_t_grid.txt"
    target_u_grid = "target_u_grid.txt"

    create_cdo_grid_descriptor(
        ncfile=first_ice_file,
        gridfile=source_t_grid,
        lon_name="TLON",
        lat_name="TLAT",
    )

    create_cdo_grid_descriptor(
        ncfile=first_ice_file,
        gridfile=source_u_grid,
        lon_name="ULON",
        lat_name="ULAT",
    )

    run_cdo(
        f"cdo griddes {REFGRID_T} > {target_t_grid}"
    )

    run_cdo(
        f"cdo griddes {REFGRID_U} > {target_u_grid}"
    )

    # -------------------------------------------------------------------------
    # Target coordinates and angle.
    # -------------------------------------------------------------------------

    with netCDF4.Dataset(REFGRID_T, "r") as nc_ref_t:
        target_tlat = clean_array(
            nc_ref_t.variables["TLAT"][:]
        )

        target_tlon = clean_array(
            nc_ref_t.variables["TLON"][:]
        )

        tmask = clean_array(
            nc_ref_t.variables["tmask"][:]
        ) > 0

    with netCDF4.Dataset(REFGRID_U, "r") as nc_ref_u:
        target_ulat = clean_array(
            nc_ref_u.variables["ULAT"][:]
        )

        target_ulon = clean_array(
            nc_ref_u.variables["ULON"][:]
        )

    with netCDF4.Dataset(REFGRID_ANGLE, "r") as nc_angle:
        target_angle = clean_array(
            nc_angle.variables["ANGLE"][:]
        )

    if target_ocean_mask.shape != target_tlat.shape:
        raise ValueError(
            "kmt dimensions do not match target grid:\n"
            f"kmt:    {target_ocean_mask.shape}\n"
            f"target: {target_tlat.shape}"
        )

    if not np.array_equal(
        target_ocean_mask,
        tmask,
    ):
        raise ValueError(
            "cice.kmt.nc and reference tmask differ."
        )

    nj, ni = target_tlat.shape

    print(
        "Target CICE grid dimensions:",
        nj,
        ni,
    )

    print(
        "Target ocean cells:",
        np.count_nonzero(target_ocean_mask),
    )

    print(
        "Target land cells:",
        np.count_nonzero(~target_ocean_mask),
    )

    # -------------------------------------------------------------------------
    # Daily output dates.
    # -------------------------------------------------------------------------

    full_daily_dates = [
        datetime.datetime(YEAR, 1, 1)
        + datetime.timedelta(days=day)
        for day in range(
            (
                datetime.datetime(YEAR + 1, 1, 1)
                - datetime.datetime(YEAR, 1, 1)
            ).days
        )
    ]

    if TEST_START is not None:
        test_start_dt = datetime.datetime.strptime(
            TEST_START,
            "%Y-%m-%d",
        )

        test_end_dt = datetime.datetime.strptime(
            TEST_END,
            "%Y-%m-%d",
        )

        daily_dates = [
            date for date in full_daily_dates
            if test_start_dt <= date <= test_end_dt
        ]

    else:
        daily_dates = full_daily_dates

    if len(daily_dates) == 0:
        raise ValueError(
            "No output daily dates selected."
        )

    print(
        "Daily output period:",
        daily_dates[0].strftime("%Y-%m-%d"),
        "to",
        daily_dates[-1].strftime("%Y-%m-%d"),
    )

    print(
        "Number of daily records:",
        len(daily_dates),
    )

    # -------------------------------------------------------------------------
    # Create output files.
    # -------------------------------------------------------------------------

    full_boundary_file = (
        f"{BC_FILE_PREFIX}.{YEAR}.nc"
    )

    nc_out = create_boundary_file(
        outfile=full_boundary_file,
        target_tlat=target_tlat,
        target_tlon=target_tlon,
        target_ulat=target_ulat,
        target_ulon=target_ulon,
    )

    # -------------------------------------------------------------------------
    # Build source record times.
    # -------------------------------------------------------------------------

    source_record_times = []

    for date_str in source_months:
        year_month = datetime.datetime.strptime(
            date_str,
            "%Y-%m",
        )

        source_record_times.append(
            month_midpoint(
                year_month.year,
                year_month.month,
            )
        )

    # -------------------------------------------------------------------------
    # Process one pair of monthly source records at a time.
    #
    # This avoids storing the complete daily year in memory.
    # -------------------------------------------------------------------------

    output_index = 0
    state_left = None

    for interval_index in range(
        len(source_months) - 1
    ):

        month_left = source_months[interval_index]
        month_right = source_months[interval_index + 1]

        time_left = source_record_times[interval_index]
        time_right = source_record_times[interval_index + 1]

        if state_left is None:
            with tempfile.TemporaryDirectory(
                prefix=f"cice_bc_{month_left}_",
            ) as workdir:
                state_left = process_month(
                    date_str=month_left,
                    source_t_grid=source_t_grid,
                    source_u_grid=source_u_grid,
                    target_t_grid=target_t_grid,
                    target_u_grid=target_u_grid,
                    target_angle=target_angle,
                    ocean_mask=target_ocean_mask,
                    workdir=workdir,
                )

        with tempfile.TemporaryDirectory(
            prefix=f"cice_bc_{month_right}_",
        ) as workdir:
            state_right = process_month(
                date_str=month_right,
                source_t_grid=source_t_grid,
                source_u_grid=source_u_grid,
                target_t_grid=target_t_grid,
                target_u_grid=target_u_grid,
                target_angle=target_angle,
                ocean_mask=target_ocean_mask,
                workdir=workdir,
            )

        interval_daily_dates = [
            date for date in daily_dates
            if time_left <= date < time_right
        ]

        if interval_daily_dates:
            print(
                "\nInterpolating interval:",
                time_left.strftime("%Y-%m-%d"),
                "to",
                time_right.strftime("%Y-%m-%d"),
            )

        interval_seconds = (
            time_right - time_left
        ).total_seconds()

        for date_now in interval_daily_dates:

            fraction = (
                (date_now - time_left).total_seconds()
                / interval_seconds
            )

            # Interpolate all state variables.
            daily_state = {}

            for key in state_left:
                daily_state[key] = interpolate_linear(
                    state_left[key],
                    state_right[key],
                    fraction,
                )

            # Derive physical daily boundary fields.
            daily_fields = derive_daily_fields(
                state=daily_state,
                ocean_mask=target_ocean_mask,
            )

            # Diagnostics.
            aice_total = daily_fields["aice_total"]

            max_aice = np.nanmax(
                aice_total[target_ocean_mask]
            )

            if max_aice > 1.000001:
                raise ValueError(
                    f"Daily sum(aicen) exceeds one: {max_aice}"
                )

            if output_index == 0 or (
                output_index % 30 == 0
            ):
                print(
                    f"Writing {date_now.strftime('%Y-%m-%d')} "
                    f"(max sum(aicen)={max_aice:.6f})"
                )

            # -------------------------------------------------------------
            # Apply target-land fill values and write daily record.
            # -------------------------------------------------------------

            aicen_write = apply_land_fill(
                daily_fields["aicen"],
                target_ocean_mask,
            )

            vicen_write = apply_land_fill(
                daily_fields["vicen"],
                target_ocean_mask,
            )

            vsnon_write = apply_land_fill(
                daily_fields["vsnon"],
                target_ocean_mask,
            )

            alvln_write = apply_land_fill(
                daily_fields["alvln"],
                target_ocean_mask,
            )

            vlvln_write = apply_land_fill(
                daily_fields["vlvln"],
                target_ocean_mask,
            )

            Tinz_write = apply_land_fill(
                daily_fields["Tinz"],
                target_ocean_mask,
            )

            Sinz_write = apply_land_fill(
                daily_fields["Sinz"],
                target_ocean_mask,
            )

            Tsnz_write = apply_land_fill(
                daily_fields["Tsnz"],
                target_ocean_mask,
            )

            Tsfc_write = apply_land_fill(
                daily_fields["Tsfc"],
                target_ocean_mask,
            )

            iage_write = apply_land_fill(
                daily_fields["iage"],
                target_ocean_mask,
            )

            apondn_write = apply_land_fill(
                daily_fields["apondn"],
                target_ocean_mask,
            )

            hpondn_write = apply_land_fill(
                daily_fields["hpondn"],
                target_ocean_mask,
            )

            ipondn_write = apply_land_fill(
                daily_fields["ipondn"],
                target_ocean_mask,
            )

            fbrine_write = apply_land_fill(
                daily_fields["fbrine"],
                target_ocean_mask,
            )

            hbrine_write = apply_land_fill(
                daily_fields["hbrine"],
                target_ocean_mask,
            )

            uvel_write = apply_land_fill(
                daily_fields["uvel"],
                target_ocean_mask,
            )

            vvel_write = apply_land_fill(
                daily_fields["vvel"],
                target_ocean_mask,
            )

            # Output time.
            nc_out.variables["time"][output_index] = (
                netCDF4.date2num(
                    date_now,
                    units=nc_out.variables["time"].units,
                    calendar=nc_out.variables["time"].calendar,
                )
            )

            nc_out.variables["Time"][output_index] = output_index

            # Category state.
            nc_out.variables["aicen"][
                output_index, :, :, :
            ] = aicen_write

            nc_out.variables["vicen"][
                output_index, :, :, :
            ] = vicen_write

            nc_out.variables["vsnon"][
                output_index, :, :, :
            ] = vsnon_write

            # Level ice.
            nc_out.variables["alvln"][
                output_index, :, :, :
            ] = alvln_write

            nc_out.variables["vlvln"][
                output_index, :, :, :
            ] = vlvln_write

            # Thermodynamics.
            nc_out.variables["Tinz"][
                output_index, :, :, :, :
            ] = Tinz_write

            nc_out.variables["Sinz"][
                output_index, :, :, :, :
            ] = Sinz_write

            nc_out.variables["Tsnz"][
                output_index, :, :, :, :
            ] = Tsnz_write

            nc_out.variables["Tsfc"][
                output_index, :, :, :
            ] = Tsfc_write

            # Age.
            nc_out.variables["iage"][
                output_index, :, :
            ] = iage_write

            # Pond and brine fields.
            nc_out.variables["apondn"][
                output_index, :, :, :
            ] = apondn_write

            nc_out.variables["hpondn"][
                output_index, :, :, :
            ] = hpondn_write

            nc_out.variables["ipondn"][
                output_index, :, :, :
            ] = ipondn_write

            nc_out.variables["fbrine"][
                output_index, :, :, :
            ] = fbrine_write

            nc_out.variables["hbrine"][
                output_index, :, :, :
            ] = hbrine_write

            # Velocity.
            nc_out.variables["uvel"][
                output_index, :, :
            ] = uvel_write

            nc_out.variables["vvel"][
                output_index, :, :
            ] = vvel_write

            output_index += 1

        # Next interval begins with the present right record.
        state_left = state_right

    nc_out.close()

    if output_index != len(daily_dates):
        raise RuntimeError(
            "Number of written daily records differs from expectation:\n"
            f"written = {output_index}\n"
            f"expected = {len(daily_dates)}"
        )

    # -------------------------------------------------------------------------
    # Produce trimmed W/E/S/N file.
    # -------------------------------------------------------------------------

    trimmed_file = (
        f"{BC_FILE_PREFIX}.trimmed.{YEAR}.nc"
    )

    make_trimmed_boundary_file(
        full_file=full_boundary_file,
        trimmed_file=trimmed_file,
    )

    KEEP_FULL_DOMAIN_FILE = False

    if not KEEP_FULL_DOMAIN_FILE:
        print(
            "Removing full-domain intermediate file:",
            full_boundary_file,
        )
    os.remove(full_boundary_file)

    print("\n" + "=" * 80)
    print("Finished successfully.")
    print("=" * 80)

    print("Full daily boundary file:")
    print(" ", full_boundary_file)

    print("Trimmed daily boundary file:")
    print(" ", trimmed_file)

    print("Written daily records:")
    print(" ", output_index)


if __name__ == "__main__":
    main()
