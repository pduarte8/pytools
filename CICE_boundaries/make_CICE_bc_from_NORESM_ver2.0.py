#!/usr/bin/env python3
import os
import glob
import sys
import subprocess
import pickle
import scipy
import numpy as np
import netCDF4
import datetime
import xarray as xr
import pandas as pd
from dateutil.relativedelta import relativedelta
from numpy import dtype
from scipy.ndimage import distance_transform_edt
# -----------------------
# Helper functions
# -----------------------
def month_midpoint_rounded_midnight(year, month):
    start = datetime.datetime(year, month, 1)
    next_month = start + relativedelta(months=1)
    midpoint = start + (next_month - start) / 2
    # round to nearest day: add 12 hours and take date, then return midnight datetime
    rounded_date = (midpoint + datetime.timedelta(hours=12)).date()
    return datetime.datetime.combine(rounded_date, datetime.time(0, 0))

def run_cdo(cmd):
    print(f"> CDO: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("CDO ERROR:", res.stderr)
        raise RuntimeError("CDO failed")
    return res

def create_cice_gridfile(ncfile, gridfile):
    ds = xr.open_dataset(ncfile)
    lon = ds["TLON"].values
    lat = ds["TLAT"].values
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
    lon = np.nan_to_num(lon, nan=1e20)
    lat = np.nan_to_num(lat, nan=1e20)
    ny, nx = lat.shape
    gridsize = nx * ny
    with open(gridfile, "w") as f:
        f.write("gridtype = curvilinear\n")
        f.write(f"gridsize = {gridsize}\n")
        f.write(f"xsize = {nx}\n")
        f.write(f"ysize = {ny}\n")
        f.write("xvals = " + " ".join(map(str, lon.flatten(order='C'))) + "\n")
        f.write("yvals = " + " ".join(map(str, lat.flatten(order='C'))) + "\n")
    ds.close()
    print(f"✔ Source grid file written: {gridfile}")

def create_cice_velocity_gridfile(ncfile, gridfile):
    ds = xr.open_dataset(ncfile)
    lon = ds["ULON"].values
    lat = ds["ULAT"].values
    lon = np.nan_to_num(lon, nan=1e20)
    lat = np.nan_to_num(lat, nan=1e20)
    ny, nx = lat.shape
    gridsize = nx * ny
    with open(gridfile, "w") as f:
        f.write("gridtype = curvilinear\n")
        f.write(f"gridsize = {gridsize}\n")
        f.write(f"xsize = {nx}\n")
        f.write(f"ysize = {ny}\n")
        f.write("xvals = " + " ".join(map(str, lon.flatten(order='C'))) + "\n")
        f.write("yvals = " + " ".join(map(str, lat.flatten(order='C'))) + "\n")
    ds.close()
    print(f"✔ Velocity grid file written: {gridfile}")

def nearest_fill_2d(field, ocean_mask):
    """
    Fill NaNs in ocean points using nearest valid ocean values.
    Land is untouched.
    """

    field = field.copy()

    # valid ocean points
    valid = ocean_mask & np.isfinite(field)

    # nothing to fill
    if np.all(valid):
        return field

    # distance transform: find nearest valid cell
    _, indices = distance_transform_edt(~valid, return_indices=True)

    missing = ocean_mask & ~np.isfinite(field)

    field[missing] = field[indices[0][missing], indices[1][missing]]

    return field


fill_value = 1.0e30

def replace_nan_with_fill(data):
    out = np.array(data, copy=True)
    out[~np.isfinite(out)] = fill_value
    return out

def create_var(name, dims):
    v = nc_out.createVariable(
        name,
        'f8',
        dims,
        zlib=False,
        fill_value=fill_value
    )
    v.missing_value = fill_value
    return v

# -----------------------
# Main processing
# -----------------------
def main():
    # -----------------------
    # Edit these parameters to your environment
    # -----------------------
    dir_NORESM_ice = '/cluster/shared/arcticfjord/NORESM/ice/'
    dir_NORESM_atm = '/cluster/shared/arcticfjord/NORESM/atm/'
    refgrid_t = '/cluster/shared/arcticfjord/hs/A4_S4K/prog_fram/008_CICE_boundary_condition_ITD_2014/iceh.2015-11.selvar.tmask.nc'
    refgrid_u = '/cluster/shared/arcticfjord/hs/A4_S4K/prog_fram/008_CICE_boundary_condition_ITD_2014/iceh.2015-11.selvar.sig2.nc'
    refgrid_angle = '/cluster/shared/arcticfjord/hs/A4_S4K/prog_fram/008_CICE_boundary_condition_ITD_2014/iceh.2015-11.selvar.ANGLE.nc'

    fname_ice_pattern = "NSSP585frc2_f09_tn14_20191105.cice.h.{date}.nc"
    fname_atm_pattern = "NSSP585frc2_f09_tn14_20191105.cam.h0.{date}.nc"

    bc_file_name = 'cice_bc_from_NORESM'
    year = 2023   # target year to create BC for (one-file-per-year)
    # -----------------------
    # Variables and constants (as in your script)
    # -----------------------
    ncat = 5
    nice_layer = 7
    nsnow_layer = 1

    scalars_2D = ["aice","hi","ardg","sirdgthick","sitempsnic","sitemptop","sitempbot","siage"]
    scalars_per_cat = ["aicen","siitdthick","siitdsnthick"]
    vectors_to_keep  = ["siu","siv"]
    scalars_2D_atm =["TS"]

    source_grid = "source_grid.txt"
    target_grid = "target_grid.txt"

    # -----------------------
    # Build months to process: year-01..year-12 plus optionally prev Dec and next Jan if files exist
    # -----------------------
    months_to_process = [f"{year}-{m:02d}" for m in range(1,13)]
    prev_dec = f"{year-1}-12"
    next_jan = f"{year+1}-01"

    prev_file = os.path.join(dir_NORESM_ice, fname_ice_pattern.format(date=prev_dec))
    next_file = os.path.join(dir_NORESM_ice, fname_ice_pattern.format(date=next_jan))
    if os.path.exists(prev_file):
        months_to_process.insert(0, prev_dec)
        print("Including previous December for interpolation:", prev_dec)
    else:
        print("Previous December not found (ok):", prev_file)
    if os.path.exists(next_file):
        months_to_process.append(next_jan)
        print("Including next January for interpolation:", next_jan)
    else:
        print("Next January not found (ok):", next_file)

    # -----------------------
    # Prepare grids: create source_grid from the first available input
    # -----------------------
    first_in = months_to_process[0]
    first_input_candidate = os.path.join(dir_NORESM_ice, fname_ice_pattern.format(date=first_in))
    if not os.path.exists(first_input_candidate):
        raise FileNotFoundError(f"Missing input for grid creation: {first_input_candidate}")
    create_cice_gridfile(first_input_candidate, source_grid)
    run_cdo(f"cdo griddes {refgrid_t} > {target_grid}")

    # read angle for final velocity rotation
    ds_angle_t = xr.open_dataset(refgrid_angle)
    angle_t = np.squeeze(ds_angle_t["ANGLE"].values)
    ds_angle_t.close()
    nc_ref = netCDF4.Dataset(refgrid_t, 'r')
    tmask = nc_ref.variables["tmask"][:]
    try:
        TLAT_ref = nc_ref.variables['TLAT'][:]
        nj, ni = TLAT_ref.shape
    finally:
        nc_ref.close()
    print("Grid dims set from refgrid_t:", nj, ni)
    # -----------------------
    # Storage for monthly-derived BC arrays
    # -----------------------
    monthly_times = []
    store = {}
    keys_4d = ['aicen','vicen','vsnon','alvln','vlvln','Tsfc']          # (time, ncat, eta_t, xi_t)
    keys_5d = ['Tinz','Sinz','Tsnz']                                 # (time, ncat, nkice, eta_t, xi_t)
    keys_3d = ['iage']                                               # (time, eta_t, xi_t)
    keys_2d = ['uvel','vvel']                                        # (time, eta_t, xi_t)
    for k in keys_4d + keys_5d + keys_3d + keys_2d + ['apondn','hpondn','ipondn','fbrine','hbrine']:
        store[k] = []

    # we will determine nj, ni on first processed month
    #nj = ni = None

    # -----------------------
    # Loop over monthly inputs: process, compute BC arrays and store them
    # -----------------------
    for date_str in months_to_process:
        infile = os.path.join(dir_NORESM_ice, fname_ice_pattern.format(date=date_str))
        print("\nProcessing monthly file:", infile)
        if not os.path.exists(infile):
            print("  SKIP - not found:", infile)
            continue

        ds = xr.open_dataset(infile)

        # get grid dims from one variable (aice) after reading
        #if nj is None:
        #    try:
        #        aice_sample = ds['aice']
        #        nj, ni = aice_sample.shape[-2], aice_sample.shape[-1]
        #        print("Grid dims detected nj,ni =", nj, ni)
        #    except Exception:
        #        raise RuntimeError("Cannot determine grid size from input.")

        # ----- STEP 1: 2D variables remap -----
        tmp_2d = f"temp_2d_{date_str}.nc"
        tmp_2d_withgrid = f"temp_2d_withgrid_{date_str}.nc"
        out_2d = f"out_2d_{date_str}.nc"
        if os.path.exists(tmp_2d): os.remove(tmp_2d)
        ds[scalars_2D].to_netcdf(tmp_2d)
        run_cdo(f"cdo setgrid,{source_grid} {tmp_2d} {tmp_2d_withgrid}")
        if os.path.exists(out_2d): os.remove(out_2d)
        run_cdo(f"cdo remapbil,{target_grid} {tmp_2d_withgrid} {out_2d}")
        ds_out_2d = xr.open_dataset(out_2d)

        # ----- STEP 2: NCAT variables per category -----
        ds_out_combined = xr.Dataset()
        for var in scalars_per_cat:
            if var not in ds.variables:
                print(f"  WARNING: {var} not in input, skipping.")
                continue
            cat_list = []
            for ncati in range(ds.sizes['nc']):
                ds_cat = ds[var].isel(nc=ncati)
                tmp_cat = f"temp_cat_{var}_{date_str}.nc"
                tmp_cat_grid = f"temp_cat_grid_{var}_{date_str}.nc"
                if os.path.exists(tmp_cat): os.remove(tmp_cat)
                ds_cat.to_netcdf(tmp_cat)
                run_cdo(f"cdo setgrid,{source_grid} {tmp_cat} {tmp_cat_grid}")
                run_cdo(f"cdo remapbil,{target_grid} {tmp_cat_grid} {tmp_cat}")
                ds_remap = xr.open_dataset(tmp_cat)
                # ds_remap[var] should have time dim; expand to include nc dim
                ds_remap_var = ds_remap[var].expand_dims({'nc':[ncati]})
                cat_list.append(ds_remap_var)
                ds_remap.close()
                # cleanup tmp_cat* will be done later
            if len(cat_list) > 0:
                combined_var = xr.concat(cat_list, dim='nc')
                # ensure ordering (time, nc, nj, ni)
                combined_var = combined_var.transpose('time','nc','nj','ni')
                ds_out_combined[var] = combined_var

        # ----- STEP 3: velocities -----
        vel_source_grid = f"source_grid_vel_{date_str}.txt"
        create_cice_velocity_gridfile(infile, vel_source_grid)
        for uvar, vvar in zip(vectors_to_keep[::2], vectors_to_keep[1::2]):
            ds_uv = xr.Dataset({uvar: ds[uvar], vvar: ds[vvar], "ANGLE": ds["ANGLE"]})
            tmp_uv = f"temp_uv_{date_str}.nc"
            tmp_uv_withgrid = f"temp_uv_withgrid_{date_str}.nc"
            if os.path.exists(tmp_uv): os.remove(tmp_uv)
            ds_uv.to_netcdf(tmp_uv)
            run_cdo(f"cdo setgrid,{vel_source_grid} {tmp_uv} {tmp_uv_withgrid}")
            run_cdo(f"cdo remapbil,{target_grid} {tmp_uv_withgrid} {tmp_uv}")
            ds_remap = xr.open_dataset(tmp_uv)
            u_interp = np.squeeze(ds_remap[uvar].values)
            v_interp = np.squeeze(ds_remap[vvar].values)
            angle_src_interp = np.squeeze(ds_remap["ANGLE"].values)
            # rotate grid-aligned to east/north, then to target grid orientation
            u_e = u_interp * np.cos(angle_src_interp) - v_interp * np.sin(angle_src_interp)
            v_n = u_interp * np.sin(angle_src_interp) + v_interp * np.cos(angle_src_interp)
            u_final =  u_e * np.cos(angle_t) + v_n * np.sin(angle_t)
            v_final = -u_e * np.sin(angle_t) + v_n * np.cos(angle_t)
            # store final fields in ds_out_combined (as 2D arrays with time dim)
            ds_out_combined[uvar] = (('time','nj','ni'), np.expand_dims(u_final, axis=0))
            ds_out_combined[vvar] = (('time','nj','ni'), np.expand_dims(v_final, axis=0))
            ds_remap.close()

        # ----- STEP 4: fill land mask and write intermediate combined_{date}.nc -----
        # merge ds_out_2d and ds_out_combined
        ds_2d_for_merge = ds_out_2d
        ds_all_for_merge = ds_out_combined
        combined_ds = xr.merge([ds_2d_for_merge, ds_all_for_merge])
        # replace NaNs with fill_value
        for var in combined_ds.data_vars:
            data = combined_ds[var].values
            #mask = np.isnan(data) | (data > 1e20)
            #data[mask] = fill_value
            # keep NaN as NaN
            fill = combined_ds[var].attrs.get('_FillValue', None)
            if fill is not None:
                data[data == fill] = np.nan
            # also catch extreme garbage values safely
            data[data > 1e20] = np.nan
            data[data < -1e20] = np.nan    
            combined_ds[var].values = data

        combined_outfile = f"combined_{date_str}.nc"
        combined_ds.to_netcdf(combined_outfile, mode="w")
        ds_out_2d.close()
        combined_ds.close()

        ocean_mask = tmask > 0
        # ----- STEP 5: atmosphere remap (2D atm->out_atm) -----
        infile_atm = os.path.join(dir_NORESM_atm, fname_atm_pattern.format(date=date_str))
        if not os.path.exists(infile_atm):
            raise FileNotFoundError(f"Missing atmosphere file: {infile_atm}")
        ds_atm = xr.open_dataset(infile_atm)
        tmp_atm = f"temp_atm_{date_str}.nc"
        tmp_atm_withgrid = f"temp_atm_withgrid_{date_str}.nc"
        out_atm = f"out_atm_{date_str}.nc"
        if os.path.exists(tmp_atm): os.remove(tmp_atm)
        ds_atm[scalars_2D_atm].to_netcdf(tmp_atm)
        run_cdo(f"cdo setgrid,{source_grid} {tmp_atm} {tmp_atm_withgrid}")
        if os.path.exists(out_atm): os.remove(out_atm)
        run_cdo(f"cdo remapbil,{target_grid} {tmp_atm_withgrid} {out_atm}")
        ds_atm_out = xr.open_dataset(out_atm)

        # ----- STEP 6: compute BC fields from combined_{date}.nc and out_atm -----
        nc = netCDF4.Dataset(combined_outfile, 'r')
        # read the first time index fields (monthly files are single-time)
        aice = nc.variables['aice'][0, :, :]
        hi = nc.variables['hi'][0, :, :]
        iceage = nc.variables['siage'][0, :, :]
        nj_read, ni_read = aice.shape

        # DIAGNOSTIC SNIPPET - paste this in the script right after `nc = netCDF4.Dataset(combined_outfile, 'r')
        print("DEBUG: combined_outfile =", combined_outfile)
        print("DEBUG: expected grid (nj,ni) =", (nj, ni))
        # Show aice var raw shape and dims
        aice_var = nc.variables['aice']
        print("DEBUG: aice var shape (raw):", getattr(aice_var, 'shape', None))
        print("DEBUG: aice var dimensions:", getattr(aice_var, 'dimensions', None))

        # read raw aice and squeeze possible time dim
        aice_raw = aice_var[:]
        print("DEBUG: aice_raw.shape before squeeze:", aice_raw.shape)
        aice_arr = np.squeeze(aice_raw)
        print("DEBUG: aice_arr.shape after squeeze:", aice_arr.shape)

        # print for hi and siage as well if present
        print("DEBUG: hi var shape/raw:", np.shape(nc.variables['hi'][:]))
        print("DEBUG: siage var shape/raw:", np.shape(nc.variables['siage'][:]))

        if (nj_read != nj) or (ni_read != ni):
            raise RuntimeError("Grid mismatch between earlier detection and current file.")
        # atmosphere t2m
        nc_t2m = netCDF4.Dataset(out_atm, 'r')
        t2m = nc_t2m.variables['TS'][:]
        t2m_now = t2m[0, :, :]
        nc_t2m.close()
        # read category variables
        aicen_arr = np.zeros((ncat, nj, ni))   # sea ice conc. of i-th category
        hin   = np.zeros((ncat, nj, ni))   # ice thickness of i-th category, [m]
        hsn   = np.zeros((ncat, nj, ni))   # snow thickness of i-th category, [m]
        ridged_fraction = np.zeros((nj, ni))

        aicen_arr = nc.variables['aicen'][0, :, :, :]     # (nc, nj, ni)

        print("aicen:", np.nanmin(aicen_arr), np.nanmax(aicen_arr))
        print("Number of values > 1:", np.sum(aicen_arr > 1))
        print("Number of values < 0:", np.sum(aicen_arr < 0))
        
        hin_arr = nc.variables['siitdthick'][0, :, :, :]
        hsn_arr = nc.variables['siitdsnthick'][0, :, :, :]
        ridged_fraction = nc.variables['ardg'][0, :, :]

        # convert fill values → NaN (if not already done upstream)
        aicen_arr = np.where(aicen_arr == fill, np.nan, aicen_arr)
        hin_arr   = np.where(hin_arr == fill, np.nan, hin_arr)
        hsn_arr   = np.where(hsn_arr == fill, np.nan, hsn_arr)
        ridged_fraction = np.where(ridged_fraction == fill, np.nan, ridged_fraction)

        # apply physical bounds (only for aicen)
        aicen_arr[(aicen_arr < 0) | (aicen_arr > 1)] = np.nan
        hin_arr[hin_arr < 0] = np.nan
        hsn_arr[hsn_arr < 0] = np.nan
        ridged_fraction[(ridged_fraction < 0) | (ridged_fraction > 1)] = np.nan
        
        for nc in range(ncat):

            aicen_arr[nc] = nearest_fill_2d(aicen_arr[nc], ocean_mask)
            hin_arr[nc]   = nearest_fill_2d(hin_arr[nc], ocean_mask)
            hsn_arr[nc]   = nearest_fill_2d(hsn_arr[nc], ocean_mask)
            aicen_arr[:, ~ocean_mask] = np.nan
            hin_arr[:, ~ocean_mask] = np.nan
            hsn_arr[:, ~ocean_mask] = np.nan

        total = np.sum(aicen_arr, axis=0)
        masktot = total > 1.0

        aicen_arr[:, masktot] /= total[masktot]  # This is to make sure that the sum of all categories is <= 1
        
        ridged_fraction = nearest_fill_2d(ridged_fraction, ocean_mask)
        ridged_fraction[~ocean_mask] = np.nan
        # initialize arrays to store BC results for this month
        vicen_arr = np.zeros_like(aicen_arr)
        vsnon_arr = np.zeros_like(aicen_arr)
        alvl_arr = np.zeros_like(aicen_arr)
        vlvl_arr = np.zeros_like(aicen_arr)
        t_ice = np.zeros((ncat, nice_layer, nj, ni))
        s_ice = np.zeros((ncat, nice_layer, nj, ni))
        tsnow = np.zeros((ncat, nsnow_layer, nj, ni))
        # t_snoice, t_ice_top, t_ice_bot left as zeros (original code assigned them earlier)
        t_snoice = np.zeros((nj, ni))
        t_ice_top = np.zeros((nj, ni))
        t_ice_bot = np.zeros((nj, ni))

        # compute derived category-level BC arrays
        for j in range(nj):
            for i in range(ni):
                if aice[j, i] > 0.01 and hi[j, i] > 0.01:
                    age = iceage[j ,i] / (3600.0 * 24.0 * 365.0)
                    for nnc in range(ncat):
                        if aicen_arr[nnc, j, i] > 0.01 and hin_arr[nnc, j, i] > 0.01:
                            vicen_arr[nnc, j, i] = hin_arr[nnc, j, i] * aicen_arr[nnc, j ,i]
                            vsnon_arr[nnc, j, i] = hsn_arr[nnc, j, i] * aicen_arr[nnc, j ,i]
                            alvl_arr[nnc, j, i] = (1.0 - ridged_fraction[j, i]) * aicen_arr[nnc, j, i]
                            vlvl_arr[nnc, j, i] = vicen_arr[nnc, j ,i] * (1.0 - ridged_fraction[j, i])
                            t2m0 = np.nanmin([t2m_now[j, i], -1.0e-5])
                            t_grad = - (t2m0 - t_snoice[j, i]) / hsn_arr[nnc, j, i] if hsn_arr[nnc, j, i] > 0 else 0.0
                            tsnow[nnc, 0, j, i] = t2m0 + t_grad * (hsn_arr[nnc, j, i] * 0.5)
                            t_grad = - (t_ice_top[j, i] - t_ice_bot[j, i]) / hin_arr[nnc, j, i] if hin_arr[nnc, j, i] > 0 else 0.0
                            for k in range(nice_layer):
                                z_ice = (hin_arr[nnc, j, i] / nice_layer) * (float(k) + 0.5)
                                t_ice[nnc, k, j, i] = t_ice_top[j, i] + t_grad * z_ice
                            for k in range(nice_layer):
                                z = ((k + 1) - 0.5) / float(nice_layer)
                                if age > 1.5:
                                    a = 0.407; b = 0.573; s_max = 3.2
                                    theta =  np.pi * z**(a/(z + b))
                                    s_ice[nnc, k, j, i] = 0.5 * s_max * (1.0 - np.cos(theta))
                                else:
                                    s_ice[nnc, k, j, i] = 19.539 * z**2 - 19.93 * z + 8.913

        # velocities: read from earlier temp_uv (we created out uv netcdf per month)
        # Use u/v arrays we stored into ds_out_combined if present, otherwise fallback to combined file
        try:
            nc_uv = netCDF4.Dataset(f"temp_uv_{date_str}.nc", 'r')
            uice = nc_uv.variables['siu'][0,:, :]
            vice = nc_uv.variables['siv'][0,:, :]
            u_fill = nc_uv.variables['siu']._FillValue
            v_fill = nc_uv.variables['siv']._FillValue
            u_fill = getattr(nc_uv.variables['siu'], '_FillValue', None)
            v_fill = getattr(nc_uv.variables['siv'], '_FillValue', None)
            if u_fill is not None:
                uice = np.where(uice == u_fill, np.nan, uice)
            if v_fill is not None:
                vice = np.where(vice == v_fill, np.nan, vice)
            uice[uice > 1e20] = np.nan
            vice[vice > 1e20] = np.nan
            uice[uice < -1e20] = np.nan
            vice[vice < -1e20] = np.nan
            uice = nearest_fill_2d(uice, ocean_mask)
            vice = nearest_fill_2d(vice, ocean_mask)
            uice[~ocean_mask] = np.nan
            vice[~ocean_mask] = np.nan

            print("uice after nearest neighbour:", np.nanmin(uice), np.nanmax(uice))
            print("vice after nearest neighbour:", np.nanmin(vice), np.nanmax(vice))
            nc_uv.close()
        except Exception:
            # fallback (should not normally happen)
            uice = np.zeros((nj, ni))
            vice = np.zeros((nj, ni))

        #nc.close()

        # store arrays for this month
        midpoint_dt = month_midpoint_rounded_midnight(int(date_str.split('-')[0]), int(date_str.split('-')[1]))
        monthly_times.append(midpoint_dt)
        store['aicen'].append(aicen_arr.copy())
        store['vicen'].append(vicen_arr.copy())
        store['vsnon'].append(vsnon_arr.copy())
        store['alvln'].append(alvl_arr.copy())
        store['vlvln'].append(vlvl_arr.copy())
        store['Tinz'].append(t_ice.copy())
        store['Sinz'].append(s_ice.copy())
        store['Tsnz'].append(tsnow.copy())
        # Tsfc: clipped t2m per category
        Tsfc_cat = np.zeros((ncat, nj, ni))
        t2m_now_clip = np.where(t2m_now < -1.0e-5, t2m_now, -1.0e-5)
        for n in range(ncat):
            Tsfc_cat[n,:,:] = t2m_now_clip
        store['Tsfc'].append(Tsfc_cat)
        store['iage'].append(iceage.copy())
        store['apondn'].append(np.zeros((ncat, nj, ni)))
        store['hpondn'].append(np.zeros((ncat, nj, ni)))
        store['ipondn'].append(np.zeros((ncat, nj, ni)))
        store['fbrine'].append(np.zeros((ncat, nj, ni)))
        store['hbrine'].append(np.zeros((ncat, nj, ni)))

        print("uice before storing:", np.nanmin(uice), np.nanmax(uice))
        print("vice beforestoring:", np.nanmin(vice), np.nanmax(vice))
        store['uvel'].append(uice.copy())
        store['vvel'].append(vice.copy())

        # cleanup many temp files created in this loop
        for fpath in [tmp_2d, tmp_2d_withgrid, out_2d, combined_outfile, tmp_atm, tmp_atm_withgrid, out_atm, f"temp_uv_{date_str}.nc", vel_source_grid]:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception:
                pass

    # end monthly loop

    if len(monthly_times) == 0:
        raise RuntimeError("No monthly data processed; aborting.")

    # -----------------------
    # Assemble ds_monthly with ordering by time
    # -----------------------
    time_monthly = np.array(monthly_times, dtype='datetime64[ns]')
    order = np.argsort(time_monthly)
    time_monthly = time_monthly[order]

    # stack arrays in the same order
    # ---------- robust stacking helper ----------
    def stack_and_order(key, expected_total_ndim):
        """
        Stack store[key] along a new time axis, reorder by 'order',
        then remove or add singleton axes so the resulting ndarray
        has expected_total_ndim dimensions.

        expected_total_ndim includes the time dimension, e.g.
        - 4 for ('time','ncat','eta_t','xi_t')
        - 5 for ('time','ncat','nkice','eta_t','xi_t')
        - 3 for ('time','eta_t','xi_t')
        """
        arr = np.stack([np.asarray(x) for x in store[key]], axis=0)   # (ntime, ...)
        # reorder in time
        arr = arr[order]

        # If too many dims, remove safe singleton axes (skip axis 0 which is time)
        # Try removing one singleton axis at a time until dims <= expected_total_ndim
        while arr.ndim > expected_total_ndim:
            removed = False
            for ax in range(1, arr.ndim):
                if arr.shape[ax] == 1:
                    arr = np.squeeze(arr, axis=ax)
                    removed = True
                    break
            if not removed:
            # no removable singleton axis found — cannot reduce dims safely
                break

        # If too few dims, insert singleton axes after time axis until match
        while arr.ndim < expected_total_ndim:
            arr = np.expand_dims(arr, axis=1)   # insert singleton after time

        return arr

    # ---------- build ds_monthly robustly ----------
    coords = {'time': time_monthly, 'ncat': np.arange(ncat), 'eta_t': np.arange(nj), 'xi_t': np.arange(ni)}
    ds_monthly = xr.Dataset(coords=coords)

    ds_monthly['aicen'] = (('time','ncat','eta_t','xi_t'), stack_and_order('aicen', 4))
    ds_monthly['vicen'] = (('time','ncat','eta_t','xi_t'), stack_and_order('vicen', 4))
    ds_monthly['vsnon'] = (('time','ncat','eta_t','xi_t'), stack_and_order('vsnon', 4))
    ds_monthly['alvln']  = (('time','ncat','eta_t','xi_t'), stack_and_order('alvln', 4))
    ds_monthly['vlvln']  = (('time','ncat','eta_t','xi_t'), stack_and_order('vlvln', 4))

    ds_monthly['Tinz']  = (('time','ncat','nkice','eta_t','xi_t'), stack_and_order('Tinz', 5))
    ds_monthly['Sinz']  = (('time','ncat','nkice','eta_t','xi_t'), stack_and_order('Sinz', 5))
    ds_monthly['Tsnz']  = (('time','ncat','nksnow','eta_t','xi_t'), stack_and_order('Tsnz', 5))

    ds_monthly['Tsfc']  = (('time','ncat','eta_t','xi_t'), stack_and_order('Tsfc', 4))
    ds_monthly['iage']  = (('time','eta_t','xi_t'), stack_and_order('iage', 3))

    ds_monthly['apondn'] = (('time','ncat','eta_t','xi_t'), stack_and_order('apondn', 4))
    ds_monthly['hpondn'] = (('time','ncat','eta_t','xi_t'), stack_and_order('hpondn', 4))
    ds_monthly['ipondn'] = (('time','ncat','eta_t','xi_t'), stack_and_order('ipondn', 4))
    ds_monthly['fbrine'] = (('time','ncat','eta_t','xi_t'), stack_and_order('fbrine', 4))
    ds_monthly['hbrine'] = (('time','ncat','eta_t','xi_t'), stack_and_order('hbrine', 4))

    ds_monthly['uvel'] = (('time','eta_t','xi_t'), stack_and_order('uvel', 3))
    ds_monthly['vvel'] = (('time','eta_t','xi_t'), stack_and_order('vvel', 3))

    # -----------------------
    # Interpolate to daily for the target year
    # -----------------------
    start_daily = pd.Timestamp(f"{year}-01-01")
    end_daily = pd.Timestamp(f"{year}-12-31")
    daily_times = pd.date_range(start=start_daily, end=end_daily, freq='D')
    daily_times_np = np.array(daily_times.to_pydatetime(), dtype='datetime64[ns]')

    # diagnostic: show monthly sample times used for interpolation
    print("Monthly times used for interpolation:", ds_monthly['time'].values)

    # ensure sorted and upsample to daily using xarray.resample (no SciPy required)

    #print("Monthly times used:", ds_monthly['time'].values)
    ds_monthly = ds_monthly.sortby('time')
    ds_daily = ds_monthly.resample(time='1D').interpolate('linear')
    ds_daily = ds_daily.sel(time=slice(start_daily.strftime('%Y-%m-%d'), end_daily.strftime('%Y-%m-%d')))
    print("Interpolated to daily: nt =", len(ds_daily['time'].values))

    # -----------------------
    # Write untrimmed daily BC netCDF (full fields)
    # -----------------------
    bry_file = bc_file_name + '.' + str(year) + '.nc'
    if os.path.exists(bry_file):
        print("Overwriting existing BC file:", bry_file)
        os.remove(bry_file)
    nc_out = netCDF4.Dataset(bry_file, 'w', format='NETCDF4')

    # dimensions
    nc_out.createDimension("TIME", None)
    nc_out.createDimension("eta_t", nj)
    nc_out.createDimension("xi_t", ni)
    nc_out.createDimension("ncat", ncat)
    nc_out.createDimension("nkice", nice_layer)
    nc_out.createDimension("nksnow", nsnow_layer)

    # domain coordinate variables
    Domain_columns = nc_out.createVariable('Domain_columns', 'i2', ('eta_t',))
    Domain_lines = nc_out.createVariable('Domain_lines', 'i2', ('xi_t',))
    NCAT = nc_out.createVariable('NCAT', 'i2', ('ncat',))
    Ice_layers = nc_out.createVariable('Ice_layers', 'i2', ('nkice',))
    Snow_layers = nc_out.createVariable('Snow_layers', 'i2', ('nksnow',))
    Domain_columns[:] = np.arange(0, nj)
    Domain_lines[:] = np.arange(0, ni)
    NCAT[:] = np.arange(0, ncat)
    Ice_layers[:] = np.arange(0, nice_layer)
    Snow_layers[:] = np.arange(0, nsnow_layer)

    # TLAT/TLON from refgrid
    nc_coord_t = netCDF4.Dataset(refgrid_t, 'r')
    TLAT = nc_out.createVariable('TLAT', 'f4', ('eta_t','xi_t'))
    TLAT[:] = nc_coord_t.variables['TLAT'][:]
    TLON = nc_out.createVariable('TLON', 'f4', ('eta_t','xi_t'))
    TLON[:] = nc_coord_t.variables['TLON'][:]
    nc_coord_t.close()

    # time
    time_var = nc_out.createVariable('time', 'f8', ('TIME',))
    time_var.standard_name = "time"
    time_var.long_name = "TIME"
    time_var.units = "days since 1900-01-01 00:00:00"
    time_var.calendar = "gregorian"
    time_var.axis = "T"
    time_var[:] = netCDF4.date2num(daily_times.to_pydatetime(), units=time_var.units, calendar=time_var.calendar)
    Time = nc_out.createVariable('Time', 'i2', ('TIME',))
    Time[:] = np.arange(len(daily_times))

    # helper to create BC var
    def create_var(name, dims):
        v = nc_out.createVariable(name, 'f8', dims, zlib=False)
        v.missing_value = 1.0e-30
        return v
    print(np.nanmin(ds_daily['aicen'].values),
      np.nanmax(ds_daily['aicen'].values))

    # create variables and write from ds_daily
    create_var('aicen', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['aicen'].values)
    create_var('vicen', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['vicen'].values)
    create_var('vsnon', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['vsnon'].values)
    create_var('alvln', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['alvln'].values)
    create_var('vlvln', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['vlvln'].values)
    create_var('Tinz', ('TIME','ncat','nkice','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['Tinz'].values)
    create_var('Sinz', ('TIME','ncat','nkice','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['Sinz'].values)
    create_var('Tsnz', ('TIME','ncat','nksnow','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['Tsnz'].values)
    create_var('Tsfc', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['Tsfc'].values)
    create_var('iage', ('TIME','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['iage'].values)
    create_var('apondn', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['apondn'].values)
    create_var('hpondn', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['hpondn'].values)
    create_var('ipondn', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['ipondn'].values)
    create_var('fbrine', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['fbrine'].values)
    create_var('hbrine', ('TIME','ncat','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['hbrine'].values)
    create_var('uvel', ('TIME','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['uvel'].values)
    create_var('vvel', ('TIME','eta_t','xi_t'))[:] = \
            replace_nan_with_fill(ds_daily['vvel'].values)
    


    #nc_out.close()
    print("Wrote daily untrimmed BC file:", bry_file)

    # -----------------------
    # Create trimmed BC file (oriented boundary variables) directly from ds_daily
    # -----------------------
    trimmed_name = bc_file_name + '.trimmed' + '.' + str(year) + '.nc'
    if os.path.exists(trimmed_name):
        os.remove(trimmed_name)
    nc_trim = netCDF4.Dataset(trimmed_name, 'w', format='NETCDF3_64BIT')

    # create dims in trimmed file
    nc_trim.createDimension('time', None)
    nc_trim.createDimension('eta_t', nj)
    nc_trim.createDimension('xi_t', ni)
    nc_trim.createDimension('ncat', ncat)
    nc_trim.createDimension('nkice', nice_layer)
    nc_trim.createDimension('nksnow', nsnow_layer)

    # write time var and domain coords (copy from untrimmed)
    time_t = nc_trim.createVariable('time', 'f8', ('time',))
    time_t.units = time_var.units
    time_t.calendar = time_var.calendar
    time_t[:] = time_var[:]
    Domain_columns = nc_trim.createVariable('Domain_columns', 'i2', ('eta_t',))
    Domain_columns[:] = np.arange(0, nj)
    Domain_lines = nc_trim.createVariable('Domain_lines', 'i2', ('xi_t',))
    Domain_lines[:] = np.arange(0, ni)

    # write oriented variables: for each var in ds_daily that has dims >=3 create _W/_E/_S/_N
    def safe_attrs_from(varname):
        return {}

    for varname in ds_daily.data_vars:
        dims = ds_daily[varname].dims
        nd = len(dims)
        data = ds_daily[varname]
        if nd < 3:
            # copy scalar/low-dim variables (rare here)
            v = nc_trim.createVariable(varname, 'f8', dims)
            try:
                v[:] = data.values
            except Exception:
                pass
            continue

        # West/East: remove xi_t dimension (last dim)
        w_data = data.isel({dims[-1]: 0})
        e_data = data.isel({dims[-1]: -1})
        # determine dims for W/E (drop xi dim)
        new_dims_we = tuple(d for d in dims if d != dims[-1])
        #vW = nc_trim.createVariable(varname + "_W_bry", 'f8', new_dims_we)
        vW = nc_trim.createVariable(
            varname + "_W_bry",
            'f8',
            new_dims_we,
            fill_value=fill_value
        )
        vW.missing_value = fill_value
        #vE = nc_trim.createVariable(varname + "_E_bry", 'f8', new_dims_we)
        vE = nc_trim.createVariable(
            varname + "_E_bry",
            'f8',
            new_dims_we,
            fill_value=fill_value
        )
        vE.missing_value = fill_value

        #vW[:] = np.asarray(w_data.values)
        #vE[:] = np.asarray(e_data.values)
        vW[:] = replace_nan_with_fill(w_data.values)
        vE[:] = replace_nan_with_fill(e_data.values)

        # South/North: remove eta_t dimension (assume it's the second-to-last or find index)
        # find index of eta-like dim (commonly 'eta_t' or dims[-2])
        eta_dim = None
        for d in dims:
            if 'eta' in d or d == dims[-2]:
                eta_dim = d
                break
        if eta_dim is None:
            eta_dim = dims[-2]
        s_data = data.isel({eta_dim: 0})
        n_data = data.isel({eta_dim: -1})
        # dims for S/N: drop the eta_dim
        new_dims_sn = tuple(d for d in dims if d != eta_dim)
        #vS = nc_trim.createVariable(varname + "_S_bry", 'f8', new_dims_sn)
        vS = nc_trim.createVariable(
            varname + "_S_bry",
            'f8',
            new_dims_sn,
            fill_value=fill_value
        )
        vS.missing_value = fill_value
        #vN = nc_trim.createVariable(varname + "_N_bry", 'f8', new_dims_sn)
        vN = nc_trim.createVariable(
            varname + "_N_bry",
            'f8',
            new_dims_sn,
            fill_value=fill_value
        )
        vN.missing_value = fill_value

        #vS[:] = np.asarray(s_data.values)
        #vN[:] = np.asarray(n_data.values)
        vS[:] = replace_nan_with_fill(s_data.values)
        vN[:] = replace_nan_with_fill(n_data.values)

    nc_trim.close()
    nc_out.close()
    print("Wrote trimmed BC file with oriented variables:", trimmed_name)

if __name__ == "__main__":
    main()
