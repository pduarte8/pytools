#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import netCDF4, datetime, glob, os, sys, pickle
import xarray as xr
import subprocess

from numpy import dtype

from dateutil.relativedelta import relativedelta

def main():
    """
    make CICE boundary condition from TOPAZ output

    NOTE:
    Following procedure is necessary before execution, otherwise an error occurs.
    - conda activate py3.5
    - module load CDO/1.9.10-intel-2021b

    NOTE:
    Consistency between input TOPAZ data and output should be checked!!!

    by Hiroshi Sumata / 2022.05.25: first test version
                      / 2022.08.23: ver0.4 test output netcdf file
                      / 2022.10.31: ver0.5 a small bug fix
                      / 2022.11.18: ver0.6 bug fixed
                      / 2022.11.28: ver0.7 bug fixed
                      / 2023.03.02: ver0.7 with implementing ice thickness distrib. test Ver.
                      / 2023.03.08: ver0.7 with ITD
    """
    # set filename paramters ----

    dir_NORESM_ice = '/cluster/shared/arcticfjord/NORESM/ice/'
    dir_NORESM_atm = '/cluster/shared/arcticfjord/NORESM/atm/'
    dir_atoms = './'
    #fname = 'TP4DAILY_start*.nc'
    fname_ice = 'NSSP585frc2_f09_tn14_20191105.cice.h.*.nc'
    fname_atm = 'NSSP585frc2_f09_tn14_20191105.cam.h0.*.nc'

    # set CICE parameters ----
    
    ncat = 5                                           # number of ice category
    div_ice_category = [0.0, 0.64, 1.39, 2.47, 4.57]   # division of ice categolies

    cat_lb = div_ice_category                          # lower bound thickness of each category 
    cat_ub = div_ice_category[1:ncat]                  # upper bound thickness of each category 
    cat_ub.append(1000.0)
    
    if len(div_ice_category) != ncat:
        print('ERROR(0): Inconsistency found between ncat and div_ice_category')
        print('          ncat =', ncat)
        print('          div_ice_category =', div_ice_category)
        sys.exit()
        
    nice_layer = 7
    nsnow_layer = 1
    
    # physical parameter ---
    
    tfp = -1.86    # freezing point of sea water [deg.C], Arrigo et al.1993,JGR 98:6929-6946
    
    #=====================================================================================
    # NOTE: CICE reference grid is prepared as follows,
    #
    # t grid : cdo selvar,tmask iceh.2015-11.nc iceh.2015-11.selvar.tmask.nc
    # uv grid: cdo selvar,sig2 iceh.2015-11.nc iceh.2015-11.selvar.sig2.nc
    # rotation angle: cdo selvar,ANGLE iceh.2015-11.nc iceh.2015-11.selvar.ANGLE.nc
    #/
    # NOTE: ANGLE is defined as the rotation angle of CICE X-coordinate relative to eastward
    #       vector (rotation from latitude line to CICE X-axis with anti-clockwise direction 
    #       is defined as positive). See Physical Oceanography Note (117) p118 for detail.
    #
    # by Hiroshi Sumata / 2022.09.14
    
    refgrid_t = '/cluster/shared/arcticfjord/hs/A4_S4K/prog_fram/008_CICE_boundary_condition_ITD_2014/iceh.2015-11.selvar.tmask.nc'  # referece for CICE T-grid 
    refgrid_u = '/cluster/shared/arcticfjord/hs/A4_S4K/prog_fram/008_CICE_boundary_condition_ITD_2014/iceh.2015-11.selvar.sig2.nc'   # referece for CICE UV-grid
    refgrid_angle = '/cluster/shared/arcticfjord/hs/A4_S4K/prog_fram/008_CICE_boundary_condition_ITD_2014/iceh.2015-11.selvar.ANGLE.nc' # reference for rotation angle of CICE grid
    
    #=====================================================================================
    # NOTE: 2m air temperature data is processed as follows,
    # (This part goes back from ver0.7 to ver0.3, by Hiroshi Sumata / 2023.03.08)
    #
    # original file used:
    #   betzy:/cluster/shared/arcticfjord/input_data/s800_cice_processed_a4_mod/
    #   era5_a4_svalbard_2015.nc
    #
    # # cdo selvar,Tair era5_a4_svalbard_2015.nc
    #   /cluster/home/hiroshi/data/era5_a4_svalbard_2015.selvar.Tair.nc
    # # cdo daymean era5_a4_svalbard_2015.selvar.Tair.nc
    #   era5_a4_svalbard_2015.selvar.Tair.daymean.nc
    # # cdo remapbil,iceh.2015-11.selvar.tmask.nc era5_a4_svalbard_2015.selvar.Tair.daymean.nc
    #   era5_a4_svalbard_2015.selvar.Tair.daymean.remapbil.iceh_tmask.nc

    #file_t2m = 'era5_a4_svalbard_2015.selvar.Tair.daymean.nc'
    #file_t2m = 'era5_a4_svalbard_yyyy.selvar.Tair.daymean.remapbil.iceh_tmask.nc'
    file_t2m = 'NSSP585frc2_f09_tn14_20191105.cam.h0.YYYY-MM.nc'

    #=====================================================================================    

    bc_file_name = 'cice_bc_from_NORESM'
    year = 2026
    
    date_start = datetime.datetime(year, 1, 1)
    date_end   = datetime.datetime(year + 1, 1, 1)  # for actual execution  

    nmonths = (date_end.year - date_start.year)
    
    # Create an array of dates separated by one month
    # Create an array of dates with only year and month using a list comprehension
    dates = [(date_start + relativedelta(months=i)).strftime('%Y-%m') 
         for i in range((date_end.year - date_start.year) * 12 + date_end.month - date_start.month)]

    dates_datetime = [(date_start + relativedelta(months=i)) for i in range((date_end.year - date_start.year) * 12 + date_end.month - date_start.month)]
    # Print the array of dates
    print(dates)
    
    #breakpoint()

    print('Data processing info. -----------------------')
    print('date_start:', date_start)
    print('date_end  :', date_end)
    print('dates =', dates)
    print('---------------------------------------------')
    
    # Define rotation angle of CICE grid for later use ----

    nc_coord_angle = netCDF4.Dataset(refgrid_angle, 'r')
    angles_east_to_x = nc_coord_angle.variables['ANGLE'][:, :]
    angles_units = nc_coord_angle.variables['ANGLE'].units
    nc_coord_angle.close()

    #===========================================================================================
    # Time loop ---
    #============================================================================================
    # =========================================================
    # USER SETTINGS (UNCHANGED)
    # =========================================================

    scalars_2D = ["aice","hi","ardg","sirdgthick","sitempsnic","sitemptop","sitempbot","siage"]
    scalars_per_cat = ["aicen","siitdthick","siitdsnthick"]
    vectors_to_keep  = ["siu","siv"]
    scalars_2D_atm =["TS"]

    fill_value = -9999  # fill value for land / masked points

    source_grid = "source_grid.txt"
    target_grid = "target_grid.txt"

    # =========================================================
    # FUNCTION: CREATE CLEAN CICE GRID FILE
    # =========================================================

    def create_cice_gridfile(ncfile, gridfile):
        ds = xr.open_dataset(ncfile)

         # Take only the first time step (CDO setgrid requires 2D coordinates)
        lon = ds["TLON"].values
        lat = ds["TLAT"].values

        # Fix longitude range (0–360 → -180–180)
        if np.nanmax(lon) > 180:
            lon = ((lon + 180) % 360) - 180

        # Remove invalid values (1e+30 etc.)
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
        print(f"✔ Source grid created: {gridfile}")

    # =========================================================
    # FUNCTION: CREATE CLEAN CICE GRID VELOCITY FILE
    # =========================================================


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
        print(f"✔ Velocity grid created: {gridfile}")

    # =========================================================
    # FUNCTION: SAFE CDO RUNNER
    # =========================================================

    def run_cdo(cmd):
        print(f"\nRunning: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print("CDO ERROR:\n", result.stderr)
            raise RuntimeError("CDO failed")

        return result

    # =================================================================================
    # FUNCTION:  Rotate CICE velocities from grid-aligned to W-E / N-S (earth-relative)
    # using provided rotation angles (radians).
    # =================================================================================
    def rotate_uv(u, v, angle):
    # Compute rotation angle from grid orientation
    # angle = arctan(dy/dx) of grid (approx)
    # For Arctic grids, simple finite differences approximation

        u_rot = u * np.cos(angle) - v * np.sin(angle)
        v_rot = u * np.sin(angle) + v * np.cos(angle)

        return u_rot, v_rot
    # =========================================================
    # CREATE GRIDS (DO THIS ONCE)
    # =========================================================

    # Source grid from FIRST CICE file
    first_file = f"{dir_NORESM_ice}NSSP585frc2_f09_tn14_20191105.cice.h.{dates[0]}.nc"
    create_cice_gridfile(first_file, source_grid)

    # Target grid (this part was already correct)
    run_cdo(f"cdo griddes {refgrid_t} > {target_grid}")


    # =========================================================
    # LOOP OVER TIME
    # =========================================================
    for date_str in dates:

        file_name = f"NSSP585frc2_f09_tn14_20191105.cice.h.{date_str}.nc"
        infile = f"{dir_NORESM_ice}{file_name}"

        print("\nProcessing:", infile)

        ds = xr.open_dataset(infile)
        # =========================================================
        # STEP 1 — 2D VARIABLES
        # =========================================================

        ds_2d = ds[scalars_2D]

        tempfile_2d = "temp_2d.nc"
        temp_with_grid = "temp_2d_withgrid.nc"

        # Remove previous files if they exist
        for f in [tempfile_2d, temp_with_grid]:
            if os.path.exists(f):
                os.remove(f)

        ds_2d.to_netcdf(tempfile_2d, mode="w")
        ds_2d.close()

        run_cdo(f"cdo setgrid,{source_grid} {tempfile_2d} {temp_with_grid}")

        outfile_2d = f"out_2d.nc"
        if os.path.exists(outfile_2d):
            os.remove(outfile_2d)

        run_cdo(f"cdo remapbil,{target_grid} {temp_with_grid} {outfile_2d}")

        ds_out = xr.open_dataset(outfile_2d)

        print("✔ 2D done →", outfile_2d)

        # =========================================================
        # STEP 2 — NCAT VARIABLES
        # =========================================================
        # Loop over category variables
        # Initialize an empty dataset to store all category variables
        # Initialize an empty dataset to store all category variables
        ds_out = xr.Dataset()
        # Print available variables in the dataset
        print("Available variables in the dataset:", list(ds.variables))
        for var in scalars_per_cat:
          print(f"\nProcessing variable: {var}")
          # Check if the variable exists in the dataset
          try:
             if var not in ds.variables:
                print(f"Warning: Variable {var} not found in the dataset. Skipping.")
                continue
          except Exception as e:
             print(f"Error accessing variable {var}: {e}")
             continue  
          cat_arrays = []  # Temporary list to store remapped data for each category
          print(f"\nProcessing {var} per category")
  
          # Loop over each ice category
          for ncati in range(ds.sizes['nc']):
             print(f"  Processing category {ncati} for variable {var}")
             ds_cat = ds[var].isel(nc=ncati)  # Select data for the current category
             temp_cat = f"temp_cat_{var}.nc"  # Temporary file for the current category
             temp_cat_grid = "temp_cat_grid_var}.nc"  # Temporary file with grid info
             # Remove previous files if they exist
             for f in [temp_cat, temp_cat_grid]:
                if os.path.exists(f):
                   os.remove(f)
             # Save the current category data to a temporary file
             print(f"Writing temporary file: {temp_cat}")
             ds_cat.to_netcdf(temp_cat, mode='w')
             print(f"Temporary file written: {temp_cat}")
             # Add source grid information
             print(f"Running CDO setgrid for {temp_cat}")
             run_cdo(f"cdo setgrid,{source_grid} {temp_cat} {temp_cat_grid}")
             # Remap to the target grid
             print(f"Running CDO remapbil for {temp_cat_grid}")
             run_cdo(f"cdo remapbil,{target_grid} {temp_cat_grid} {temp_cat}")
             # Load the remapped data and append it to the list
             print(f"Loading remapped data from {temp_cat}")
             ds_remap = xr.open_dataset(temp_cat)
             ds_remap_with_nc = ds_remap[var].expand_dims({'nc': [ncati]})  # Add the `nc` dimension
             cat_arrays.append(ds_remap_with_nc)
             ds_remap.close()
          # Combine all categories into a single variable
          combined_var = xr.concat(cat_arrays, dim='nc')
          # Reorder dimensions to ensure time, nc, nj, ni
          combined_var = combined_var.transpose('time', 'nc', 'nj', 'ni')
          # Add the combined variable to the output dataset
          if var in ds_out:
             print(f"Warning: Variable {var} already exists in ds_out. Overwriting.")
          ds_out[var] = combined_var
          # Debugging: Print all variables in ds_out
        print("\nVariables in ds_out after processing scalars_per_cat:")
        print(list(ds_out.data_vars)) 
        # Write the combined dataset to a single NetCDF file
        output_file = "temp_scalars_per_cat.nc"
        print(f"\nWriting all category variables to {output_file}")
        ds_out.to_netcdf(output_file)
        print(f"✔ All category variables written to {output_file}")

        # =======================
        # STEP 3 — velocities
        # =======================
        angle = ds["ANGLE"].values
        vel_source_grid = "source_grid_vel.txt"
        create_cice_velocity_gridfile(first_file, vel_source_grid)
         
        ds_angle_t = xr.open_dataset(refgrid_angle)
        angle_t = np.squeeze(ds_angle_t["ANGLE"].values)
        ds_angle_t.close()
  
        for uvar, vvar in zip(vectors_to_keep[::2], vectors_to_keep[1::2]):

            print(f"\nProcessing velocity pair: {uvar}, {vvar}")

            # ----------------------------
            # Prepare dataset (NO rotation yet)
            # ----------------------------
            ds_uv = xr.Dataset({
                uvar: ds[uvar],
                vvar: ds[vvar],
                "ANGLE": ds["ANGLE"]
            })

            temp_uv = "temp_uv.nc"
            temp_uv_withgrid = "temp_uv_withgrid.nc"

            for f in [temp_uv, temp_uv_withgrid]:
                if os.path.exists(f):
                    os.remove(f)

            ds_uv.to_netcdf(temp_uv, mode="w")
            ds_uv.close()

            # ----------------------------
            # Apply velocity source grid (U-grid)
            # ----------------------------
            run_cdo(f"cdo setgrid,{vel_source_grid} {temp_uv} {temp_uv_withgrid}")

            # ----------------------------
            # Remap everything together
            # ----------------------------
            run_cdo(f"cdo remapbil,{target_grid} {temp_uv_withgrid} {temp_uv}")

            ds_remap = xr.open_dataset(temp_uv)

            # ----------------------------
            # Extract interpolated fields
            # ----------------------------
            u_interp = np.squeeze(ds_remap[uvar].values)
            v_interp = np.squeeze(ds_remap[vvar].values)
            angle_src_interp = np.squeeze(ds_remap["ANGLE"].values)

            # ----------------------------
            # Step 1: source grid → earth
            # ----------------------------
            u_e = u_interp * np.cos(angle_src_interp) - v_interp * np.sin(angle_src_interp)
            v_n = u_interp * np.sin(angle_src_interp) + v_interp * np.cos(angle_src_interp)

            # ----------------------------
            # Step 2: earth → target grid  <--- NEW FINAL STEP
            # ----------------------------
            u_final =  u_e * np.cos(angle_t) + v_n * np.sin(angle_t)
            v_final = -u_e * np.sin(angle_t) + v_n * np.cos(angle_t)

            # ----------------------------
            # Store final velocities
            # ----------------------------
            ds_out[uvar] = (ds_remap[uvar].dims[-2:], u_final)
            ds_out[vvar] = (ds_remap[vvar].dims[-2:], v_final)

            ds_remap.close()

        
        # ============================
        # STEP 4 — Fill land mask
        # ===========================
        for var in ds_out.data_vars:
            data = ds_out[var].values
            mask = np.isnan(data) | (data > 1e20)
            data[mask] = fill_value
            ds_out[var].values = data

        encoding = {var: {'_FillValue': fill_value} for var in ds_out.data_vars}
        outfile = f"out_all_{date_str}.nc"
        # Fix time encoding issues
        if "time" in ds_out:
            ds_out["time"].attrs.pop("units", None)
            ds_out["time"].attrs.pop("calendar", None)
        ds_out.to_netcdf(outfile, encoding=encoding)
        ds_out.close()



        # =========================================================
        # STEP 5 — Combine all results into a single file
        # =========================================================
        # Paths to the intermediate files
        outfile_2d = "out_2d.nc"
        # Load the 2D variables and category variables
        ds_2d = xr.open_dataset(outfile_2d)
        ds_all = xr.open_dataset(outfile)
        # Merge the datasets
        combined_ds = xr.merge([ds_2d, ds_all])
        # Define the output file name for the combined dataset
        combined_outfile = f"combined_{date_str}.nc"
        # Write the combined dataset to a single NetCDF file
        print(f"\nWriting combined dataset to {combined_outfile}")
        combined_ds.to_netcdf(combined_outfile, mode="w")
        print(f"✔ Combined dataset written to {combined_outfile}")
        # Close the datasets to free memory
        ds_2d.close()
        ds_all.close()
        combined_ds.close()
        # Optional: Remove intermediate files to save disk space
        os.remove(outfile_2d)
        os.remove(outfile)
        
        # =========================================================
        # STEP 6 — Read and regrid NorESM monthly atm files
        # =========================================================
        infile_atm = f"{dir_NORESM_atm}NSSP585frc2_f09_tn14_20191105.cam.h0.{date_str}.nc"
        print('infile_atm = ',infile_atm)
        temp_atm = "temp_atm.nc"
        temp_atm_withgrid = "temp_atm_withgrid.nc"

        ds = xr.open_dataset(infile_atm)

        # Remove previous files if they exist
        for f in [temp_atm, temp_atm_withgrid]:
            if os.path.exists(f):
                os.remove(f)

        ds_atm = ds[scalars_2D_atm]
        ds_atm.to_netcdf(temp_atm, mode="w")
        ds_atm.close()

        run_cdo(f"cdo setgrid,{source_grid} {temp_atm} {temp_atm_withgrid}")

        out_atm = f"out_atm.nc"
        if os.path.exists(outfile_2d):
            os.remove(out_atm)
        

        run_cdo(f"cdo remapbil,{target_grid} {temp_atm_withgrid} {out_atm}")

        ds_out = xr.open_dataset(out_atm)

        print("✔ 2D done →", out_atm)


        # =========================================================
        # STEP 7 — Read values 
        # =========================================================
        nc = netCDF4.Dataset(combined_outfile, 'r')
        
        nctime = nc.variables['time'][:]
        t_cal = nc.variables['time'].calendar
        t_unit = nc.variables['time'].units

        aice = nc.variables['aice'][0, :, :]
        hi   = nc.variables['hi'][0, :, :] 
        iceage = nc.variables['siage'][0, :, :] 
    
        nj = aice.shape[0]  # nj = 454 for A4_S4K setup
        ni = aice.shape[1]  # ni = 696 for A4_S4K setup
        print('nj =', nj)
        print('ni =', ni)

        tstamp = netCDF4.num2date(nctime, units = t_unit, calendar = t_cal)[0]
        print('infile =', infile)
        print('tstamp =', tstamp)
        #nc.close()
        #breakpoint()
        nc_t2m = netCDF4.Dataset(out_atm, 'r')
        t2m = nc_t2m.variables['TS'][:]                  # t2m.shape = (365, 454, 696), "Surface temperature (radiative)" ;
        t2m_nctime = nc_t2m.variables['time'][:]
        t2m_t_unit = nc_t2m.variables['time'].units
        t2m_t_cal = nc_t2m.variables['time'].calendar    # NorESM uses a non-leap year calendar but here I am using monthly
                                                         # atmosphere results as well as monthly sea ice results. 
                                                         # Therefore, I need to match only months between atmosphere and ice files
                                                         # and I do not need to check the time stamps or time bounds
        
        #t2m_tstamps = netCDF4.num2date(t2m_nctime, units = t2m_t_unit, calendar = t2m_t_cal)
        
        #for (n, t2m_tstamp) in enumerate(t2m_tstamps):
        #    if t2m_tstamp.year == tstamp.year and \
        #       t2m_tstamp.month == tstamp.month and \
        #       t2m_tstamp.day == tstamp.day:
        #        index = n
        #        break
                
        #print('@1: index =', index, ', t2m_stamps[n] =', t2m_tstamps[n])        
        t2m_now = t2m[0, :, :]   # unit [deg.C]

        #breakpoint()

        ## check ----- [OK]    
        #print('t2m_tstamps[index] =', t2m_tstamps[index])
        #print('------------------------------------')

        ## check consistency of time boundas ----[OK]
        #t2m_tstamp_bnds = netCDF4.num2date(t2m_tbounds, units = t2m_t_unit, calendar = t2m_t_cal)
        #print('t2m_tstamp_bnds[0] =', t2m_tstamp_bnds[0])

        nc_t2m.close()      

        # consistency check ----

        if t2m.shape[1] != nj or t2m.shape[2] != ni:
            print('ERROR(1): Inconsistency found in grid size.')
            print('          Input ice data and atmospheric data should have the same horizontal dimension.')
            print('          fice.shape =', fice.shape)
            print('          t2m.shape  =', t2m.shape) 
            sys.exit()
        #breakpoint() 
        # (5) make boundary condition for CICE ----

        # (5.1) ice category classification ----
        #
        # NOTE: div_ice_category gives lower bound of each ice category, 
        #       i.e., the first category represents ice thickness from 0.0 to 0.64 m.

        # Define index of ice category ---
        # 
        # NOTE: The index starts from 0, i.e., 0, 1, 2, 3, 4 for 5 categories.
        #       If ice is thinner than div_ice_category[0], then the initialized value applied,
        #       ie., icat = 0.
        #
        # NOTE2: Since 'icat' was used to put all ice into one category in each cell, it is no  
        #        longer used when ice is distributed into each category in accordance with
        #        prescribed ice thickness distribution. Hiroshi Sumata / 2023.03.02
        #
                        
        # (5.2) define variables necessary for CICE boundary conditions ----
        #
        # See Physical Oceanography Note (118) p117-119 for description regarding the prescribed
        # ice thickness distribution. Hiroshi Sumata / 2023.03.06
        #
        # IMPORTANT NOTE ---
        #
        # vicen[n, j, i] gives ice thickness in n-th category when all the ice in the n-th cagetory
        # if equally distributed in the grid cell, i.e., fraction of the n-th category is already
        # taken into account. Therefore, vicen[n, j, i] does not directly give the representing
        # thickness of n-th category.
        # See CICE_experiments.odp, slide No.2 & 3 for description.
        #
        # by Hiroshi Sumata / 2023.03.07
        
        aicen = np.zeros((ncat, nj, ni))   # sea ice conc. of i-th category
        vicen = np.zeros((ncat, nj, ni))   # sea ice volume of i-th category, [m]
        vsnon = np.zeros((ncat, nj, ni))   # snow volume on i-th category
        hin   = np.zeros((ncat, nj, ni))   # ice thickness of i-th category, [m]
        hsn   = np.zeros((ncat, nj, ni))   # snow thickness of i-th category, [m]
        ridged_fraction = np.zeros((nj, ni))

        t_snoice = np.zeros((nj, ni))         # Snow-ice interface temperature
        t_ice_top = np.zeros((nj, ni))       # Temperature at ice top
        t_ice_bot = np.zeros((nj, ni))       # Temperature at ice bottom


        t_ice = np.zeros((ncat, nice_layer, nj, ni))  # sea ice temperature in n-th layer, i-th category
        s_ice = np.zeros((ncat, nice_layer, nj, ni))  # sea ice salinity in n-th layer, i-th category
        tsnow = np.zeros((ncat, nsnow_layer, nj, ni)) # snow temperature in m-th layer, i-th category

        alvl = np.zeros((ncat, nj, ni))               # area fraction of level ice in each category
        vlvl = np.zeros((ncat, nj, ni))               # volume fraction of level ice in each category

        # (5.3) define values of each variables ---

        aicen = nc.variables['aicen'][0, :, :, :]     # !CAUTION!! off-set exist, fice.shape = (454, 696)
        hin   = nc.variables['siitdthick'][0, :, :, :]
        hsn   = nc.variables['siitdsnthick'][0, :, :, :]
        ridged_fraction = nc.variables['ardg'][0, :, :]
        #nc.close()
        #breakpoint()
 
        for j in range(nj):
            for i in range(ni):
                
                if aice[j, i] > 0.01 and hi[j, i] > 0.01: # existence of ice follows NorESM simulation
                    age = iceage[j ,i] / (3600.0 * 24.0 * 365.0) #Converting ice age from seconds to years

                    for nnc in range(ncat):

                        if aicen[nnc, j, i] > 0.01 and hin[nnc, j, i] > 0.01:

                            vicen[nnc, j, i] = hin[nnc, j, i] * aicen[nnc, j ,i]
                            vsnon[nnc, j, i] = hsn[nnc, j, i] * aicen[nnc, j ,i]

                            #alvl[nnc, j, i] = level_frac_1[month_id, nnc] * fice[j, i]
                        
                            #vlvl[nnc, j, i] = rep_thick_1[month_id, nnc] * alvl[nnc, j, i]

                            alvl[nnc, j, i] = (1.0 - ridged_fraction[j, i]) * aicen[nnc, j, i]   # I am assuming that all categories have the same fractions of level and ridged ice, 
                                                                                                 # since in NorESM output there are no category-resolved variables for ice ridged or level fractions

                            vlvl[nnc, j, i] = vicen[nnc, j ,i] * (1.0 - ridged_fraction[j, i])

                            # define snow temperature at the middle snow layer ----
                            # NOTE: modification is necessary when more than 1 snow layer is used
                            #
                            # t2m_now[j, i] : 2m air temperature
                            # tfp           : freezng point of seawater [deg.C]
                            # t2m0          : atmospheric temperature if it is lower than freezing temp.
                            # t_grad        : temperature gradient in (ice + snow) layer, (z inceases downward)
                            #                 See Physical Oceanography Note (117) p66.
                            #
                            # NOTE:
                            # In the following calculations, snow and ice thickness should be given by actual 
                            # thickness of each, not by the thickness per unit area (e.g., vicen) but the thickness 
                            # normalized by the area actually covered with sea ice (e.g. vicen/aicen). 
                            # Since siitdthick and siitdsnthick are defined as ice thickness and snow thickness
                            # in NorESM output files, I suppose they are the same as vicen/aicen and vsnon/aicen
                            # respectively.


                            # Here snow temperature is calculated from the temperature gradient between the atmosphere
                            # and the snow-ice interface temperature. These calculations are done per category but the results
                            # should be the same for all categories since the gradient changes according to category thickness

                            t2m0 = np.nanmin([t2m_now[j, i], -1.0e-5])  # keep snow surf. temp negative
                            t_grad = - (t2m0 - t_snoice[j, i]) / hsn[nnc, j, i]
                            tsnow[nnc, 0, j, i] = t2m0 + t_grad * (hsn[nnc, j, i] * 0.5)

                            # define temperature inside the ice ----
                            # ATTENTION: index of the ice layers is numbered from top to bottom, 
                            #            i.e., the top layer is k = 1 (zero in python indexing) and
                            #            the bottom layer is k = nice_layer ((nice_layer - 1) in python)
                            #
                            # z_ice         : vertical coordinate of temperature point in ice
                            #                 (zero is at the top of snow)
                            t_grad = - (t_ice_top[j, i] - t_ice_bot[j, i]) / hin[nnc, j, i]
                            for k in range(nice_layer):
                                z_ice = (hin[nnc, j, i] / nice_layer) * (float(k) + 0.5)
                                t_ice[nnc, k, j, i] = t_ice_top[j, i] + t_grad * z_ice
                        
                            # define salinity inside the ice ----
                            # The vertical profile of salinity in sea ice is given by an empirical equation
                            # based on Gerland et al. (1999). See relevant description in CRiceS_tips.html
                            #
                            for k in range(nice_layer):
                    
                                z = ((k + 1) - 0.5) / float(nice_layer)  # NOTE indexing starts from zero
                            
                                if age > 1.5:  # multi-year ice
                                    a = 0.407
                                    b = 0.573
                                    s_max = 3.2                       # maximum salinity in the ice

                                    theta =  np.pi * z**(a/(z + b))
                                    s_ice[nnc, k, j, i] = 0.5 * s_max * (1.0 - np.cos(theta))
                                else:                                 # FYI
                                    s_ice[nnc, k, j, i] = 19.539 * z**2 - 19.93 * z + 8.913
                            
        #breakpoint()        
        

        uice = nc.variables['siu'][:, :]     # !CAUTION!! off-set exist, fice.shape = (454, 696)
        vice = nc.variables['siv'][:, :]     # !CAUTION!! off-set exist, hice.shape = (454, 696)
        uice_missing_value = nc.variables['siu']._FillValue
        vice_missing_value = nc.variables['siv']._FillValue
        
        nj_uv = uice.shape[0]  # nj = 454 for A4_S4K setup
        ni_uv = uice.shape[1]  # ni = 696 for A4_S4K setup

        # check ---

        if nj_uv != nj or ni_uv != ni:
            sys.exit('Inconsistency found between T and UV cells')
        
        tstamp_uv = netCDF4.num2date(nctime, units = t_unit, calendar = t_cal)[0]
        #print('infile =', infile)
        #print('tstamp =', tstamp)
        #print('uice.shape =', uice.shape)
        #print('vice.shape =', vice.shape)
        
        nc.close()                            

        #breakpoint()
        #=========================================================================================
        #
        # prepare netcdf output when processing the first day of the year
        #
        #=========================================================================================
                            
        if date_str == dates[0]: 

            bry_file = bc_file_name + '.' + str(year) + '.nc'
            nc_out = netCDF4.Dataset(bry_file, 'w', format = 'NETCDF4')

            nc_out.createDimension("TIME", None)
            nc_out.createDimension("eta_t", nj)
            nc_out.createDimension("xi_t", ni)
            nc_out.createDimension("ncat", ncat)
            nc_out.createDimension("nkice", nice_layer)
            nc_out.createDimension("nksnow", nsnow_layer)

            # create variables ---- Domain_columns, Domain_lines, NCAT, Ice_layers, Snow_layers

            Domain_columns = nc_out.createVariable('Domain_columns', dtype('int16').char, ('eta_t'))
            Domain_lines = nc_out.createVariable('Domain_lines', dtype('int16').char, ('xi_t'))
            NCAT = nc_out.createVariable('NCAT', dtype('int16').char, ('ncat'))
            Ice_layers = nc_out.createVariable('Ice_layers', dtype('int16').char, ('nkice'))            
            Snow_layers = nc_out.createVariable('Snow_layers', dtype('int16').char, ('nksnow'))
            
            Domain_columns[:] = range(0, nj)
            Domain_lines[:] = range(0, ni)
            NCAT[:] = range(0, ncat)
            Ice_layers[:] = range(0, nice_layer)
            Snow_layers[:] = range(0, nsnow_layer)

            # create variables ---- TLAT, TLON

            TLAT = nc_out.createVariable('TLAT', dtype('float32').char, ('eta_t', 'xi_t'))
                                
            nc_coord_t = netCDF4.Dataset(refgrid_t, 'r')
            TLAT[:, :] = nc_coord_t.variables['TLAT'][:, :]
            TLAT.standard_name = nc_coord_t.variables['TLAT'].standard_name
            TLAT.long_name = nc_coord_t.variables['TLAT'].long_name
            TLAT.units = nc_coord_t.variables['TLAT'].units
            TLAT._CoordinateAxisType = nc_coord_t.variables['TLAT']._CoordinateAxisType

            TLON = nc_out.createVariable('TLON', dtype('float32').char, ('eta_t', 'xi_t'))            
            TLON[:, :] = nc_coord_t.variables['TLON'][:, :]
            TLON.standard_name = nc_coord_t.variables['TLON'].standard_name
            TLON.long_name = nc_coord_t.variables['TLON'].long_name
            TLON.units = nc_coord_t.variables['TLON'].units
            TLON._CoordinateAxisType = nc_coord_t.variables['TLON']._CoordinateAxisType

            nc_coord_t.close()

            # create variable ---- time
            #
            # NOTE: definition of time should be changed if it doesn't work!
            #       ---> Should this be consistent with the hours since ??, used in BRY_YYYY.nc?
            
            time = nc_out.createVariable('time', dtype('float64').char, ('TIME')) # time stamp
            time.standard_name = "time"
            time.long_name = "TIME"
            time.units = "days since 1900-01-01 00:00:00"
            time.calendar = "gregorian"
            time.axis = "T"
            
            time[:] = netCDF4.date2num(dates_datetime[:], time.units, calendar = time.calendar)
            print('dates[0] =', dates[0], ',  time[0] =', time[0])

            Time = nc_out.createVariable('Time', dtype('int16').char, ('TIME')) # int numbers
            
            # create variable ---- aicen
            
            aicen_d = nc_out.createVariable('aicen', dtype('float64').char,
                                            ('TIME', 'ncat', 'eta_t', 'xi_t'))
            aicen_d.long_name     = 'total concenration of ice in grid cell (in category n'
            aicen_d.units         = '0-1'
            aicen_d.missing_value = 1.0e-30
            aicen_d.coordinates = "TLON TLAT NCAT time"

            # create variable ---- vicen
            
            vicen_d = nc_out.createVariable('vicen', dtype('float64').char,
                                            ('TIME', 'ncat', 'eta_t', 'xi_t'))
            vicen_d.long_name     = 'ice volume per unit area of ice (in ctegory n)'
            vicen_d.units         = 'm'
            vicen_d.missing_value = 1.0e-30
            vicen_d.coordinates = "TLON TLAT NCAT time"
            
            # create variable ---- vsnon

            vsnon_d = nc_out.createVariable('vsnon', dtype('float64').char,
                                            ('TIME', 'ncat', 'eta_t', 'xi_t'))
            vsnon_d.long_name     = 'volume per unit area of snow (in category n)'
            vsnon_d.units         = 'm'
            vsnon_d.missing_value = 1.0e-30
            vsnon_d.coordinates = "TLON TLAT NCAT time"            

            # create variable ---- alvl

            alvl_d = nc_out.createVariable('alvln', dtype('float64').char,
                                           ('TIME', 'ncat', 'eta_t', 'xi_t'))
            alvl_d.longname      = 'Areal fraction of level ice (in category n)'
            alvl_d.units         = '0-1'
            alvl_d.missing_value = 1.0e-30
            alvl_d.coordinates = "TLON TLAT NCAT time"            
            
            # create variable ---- vlvl

            vlvl_d = nc_out.createVariable('vlvln', dtype('float64').char,
                                           ('TIME', 'ncat', 'eta_t', 'xi_t'))
            vlvl_d.long_name = 'Volume fraction of level ice (in category n)'
            vlvl_d.units     = 'm?'
            vlvl_d.missing_value = 1.0e-30
            vlvl_d.coordinates = "TLON TLAT NCAT time"                        

            # create variable ---- Tinz

            Tinz = nc_out.createVariable('Tinz', dtype('float64').char,
                                         ('TIME', 'ncat', 'nkice', 'eta_t', 'xi_t'))
            Tinz.longname      = 'Tempererature in ice layer (in category n)'
            Tinz.units         = 'deg. C'
            Tinz.missing_value = 1.0e-30
            Tinz.coordinates = "TLON TLAT Ice_layers NCAT time"

            # create variable ---- Sinz

            Sinz = nc_out.createVariable('Sinz', dtype('float64').char,
                                         ('TIME', 'ncat', 'nkice', 'eta_t', 'xi_t'))
            Sinz.longname      = 'Salinity in ice layer (in category n)'
            Sinz.units         = 'psu'
            Sinz.missing_value = 1.0e-30
            Sinz.coordinates = "TLON TLAT Ice_layers NCAT time"

            # create variable ---- Tsnz

            Tsnz = nc_out.createVariable('Tsnz', dtype('float64').char,
                                         ('TIME', 'ncat', 'nksnow', 'eta_t', 'xi_t'))
            Tsnz.longname      = 'Temperature in snow layer (in category n)'
            Tsnz.units         = 'deg. C'
            Tsnz.missing_value = 1.0e-30
            Tsnz.coordinates = "TLON TLAT Snow_layers NCAT time"

            # create variable ---- Tsfc

            Tsfc = nc_out.createVariable('Tsfc', dtype('float64').char,
                                         ('TIME', 'ncat', 'eta_t', 'xi_t'))
            Tsfc.longname      = 'Ice surface temperature'
            Tsfc.units         = 'deg. C'
            Tsfc.missing_value = 1.0e-30
            Tsfc.coordinates = "TLON TLAT NCAT time"

            # create variable ---- iage

            iage = nc_out.createVariable('iage', dtype('float64').char,
                                         ('TIME', 'eta_t', 'xi_t'))
            iage.longname      = 'age of first year ice'
            iage.units         = 'day'
            iage.missing_value = 1.0e-30
            iage.coordinates   = "TLON TLAT time"
            
            # create varialbe ---- apondn

            apondn = nc_out.createVariable('apondn', dtype('float64').char,
                                           ('TIME', 'ncat', 'eta_t', 'xi_t'))
            apondn.longname      = 'melt pond fraction'
            apondn.units         = '0-1'
            apondn.missing_value = 1.0e-30
            apondn.coordinates   = "TLON TLAT NCAT time"
            
            # create varialbe ---- hpondn

            hpondn = nc_out.createVariable('hpondn', dtype('float64').char,
                                           ('TIME', 'ncat', 'eta_t', 'xi_t'))
            hpondn.longname      = 'local pond depth'
            hpondn.units         = 'm'
            hpondn.missing_value = 1.0e-30
            hpondn.coordinates   = "TLON TLAT NCAT time"
            
            # create varialbe ---- ipondn

            ipondn = nc_out.createVariable('ipondn', dtype('float64').char,
                                           ('TIME', 'ncat', 'eta_t', 'xi_t'))
            ipondn.longname      = 'mean pond ice thickness over sea ice'
            ipondn.units         = 'm'
            ipondn.missing_value = 1.0e-30
            ipondn.coordinates   = "TLON TLAT NCAT time"
            
            # create variable ---- fbrine

            fbrine = nc_out.createVariable('fbrine', dtype('float64').char,
                                           ('TIME', 'ncat', 'eta_t', 'xi_t'))
            fbrine.longname      = 'unknown'
            fbrine.units         = 'unknown'
            fbrine.missing_value = 1.0e-30
            fbrine.coordinates   = "TLON TLAT NCAT time"
                                
            # create variable ---- hbrine

            hbrine = nc_out.createVariable('hbrine', dtype('float64').char,
                                           ('TIME', 'ncat', 'eta_t', 'xi_t'))
            hbrine.longname      = 'brine surface height above ice base' 
            hbrine.units         = 'm'
            hbrine.missing_value = 1.0e-30
            hbrine.coordinates   = "TLON TLAT NCAT time"            

            #====================================================================================
            # variables associated with sea ice velocity
            #====================================================================================
            # create variables ----- ULAT, ULON

            nc_coord_u = netCDF4.Dataset(refgrid_u, 'r')
            
            ULAT = nc_out.createVariable('ULAT', dtype('float32').char, ('eta_t', 'xi_t'))
            ULAT[:, :]               = nc_coord_u.variables['ULAT'][:, :]
            ULAT.standard_name       = nc_coord_u.variables['ULAT'].standard_name
            ULAT.long_name           = nc_coord_u.variables['ULAT'].long_name
            ULAT.units               = nc_coord_u.variables['ULAT'].units
            ULAT._CoordinateAxisType = nc_coord_u.variables['ULAT']._CoordinateAxisType

            ULON = nc_out.createVariable('ULON', dtype('float32').char, ('eta_t', 'xi_t'))
            ULON[:, :]               = nc_coord_u.variables['ULON'][:, :]
            ULON.standard_name       = nc_coord_u.variables['ULON'].standard_name
            ULON.long_name           = nc_coord_u.variables['ULON'].long_name
            ULON.units               = nc_coord_u.variables['ULON'].units
            ULON._CoordinateAxisType = nc_coord_u.variables['ULON']._CoordinateAxisType            

            nc_coord_u.close()

            # create variables ---- uvel, vvel
            #
            # NOTE: horizontal dimensions of uvel and vvel are given by eta_t and xi_t, so as to
            #       mimic old BRY_yyyy.nc data, though it should be formally 'eta_u' and 'xi_u'.
            template_file = "temp_uv.nc"
            nc2 = netCDF4.Dataset(template_file, 'r')        
            nc2.set_auto_mask(False)           # by Hiroshi Sumata / 2022.11.28
            
            uvel = nc_out.createVariable('uvel', dtype('float64').char, ('TIME', 'eta_t', 'xi_t'))
            uvel.standard_name = nc2.variables['siu'].long_name
            uvel.units         = nc2.variables['siu'].units
            uvel.coordinates   = nc2.variables['siu'].coordinates
            uvel.missing_value = nc2.variables['siu'].missing_value
            uvel.cell_methods  = nc2.variables['siu'].cell_methods

            vvel = nc_out.createVariable('vvel', dtype('float64').char, ('TIME', 'eta_t', 'xi_t'))
            vvel.standard_name = nc2.variables['siv'].long_name
            vvel.units         = nc2.variables['siv'].units
            vvel.coordinates   = nc2.variables['siv'].coordinates
            vvel.missing_value = nc2.variables['siv'].missing_value
            vvel.cell_methods  = nc2.variables['siv'].cell_methods            

            nc2.close() 
        # write data into netcdf output ----
        ndate = dates.index(date_str)
        Time[ndate] = ndate
                                
        aicen_d[ndate, :, :, :] = aicen[:, :, :]
        vicen_d[ndate, :, :, :] = vicen[:, :, :]
        vsnon_d[ndate, :, :, :] = vsnon[:, :, :]
        alvl_d[ndate, :, :, :] = alvl[:, :, :]
        vlvl_d[ndate, :, :, :] = vlvl[:, :, :]
        Tinz[ndate, :, :, :, :] = t_ice[:, :, :, :]
        Sinz[ndate, :, :, :, :] = s_ice[:, :, :, :]
        Tsnz[ndate, :, :, :, :] = tsnow[:, :, :, :]
        
        for n in range(ncat):
            t2m_now = np.where(t2m_now < -1.0e-5, t2m_now, -1.0e-5) # don't provide postive temp. at ice surf.
            Tsfc[ndate, n, :, :] = t2m_now[:, :]


        iage[ndate, :, :] = iceage[:, :]
        apondn[ndate, :, :, :] = 0.0
        hpondn[ndate, :, :, :] = 0.0
        ipondn[ndate, :, :, :] = 0.0        
        fbrine[ndate, :, :, :] = 0.0
        hbrine[ndate, :, :, :] = 0.0

        uvel[ndate, :, :] = uice[:, :]
        vvel[ndate, :, :] = vice[:, :]

    nc_out.close()

    #=========================================================================================
    #
    # Make trimmed Boundary Condition
    #
    # NOTE: BC before trimming is also written, since it may be used to check spatial pattern
    #       of input data
    #
    # by Hiroshi Sumata / 2022.09.23
    #=========================================================================================

    # f : boundary condition netcdf file without trimming
    # nc: trimmed BC
    
    bry_file = bc_file_name + '.' + str(year) + '.nc'
    
    f = netCDF4.Dataset(bry_file, 'r')

    nc_name = bc_file_name + '.trimmed' + '.' + str(year) + '.nc'
    nc = netCDF4.Dataset(nc_name, 'w', format = 'NETCDF3_64BIT')

    # create dimension ----
    
    for i in f.dimensions.keys():
        nc.createDimension(i, (len(f.dimensions[i]) if not f.dimensions[i].isunlimited() else None))

    for name, variable in f.variables.items():

        # create coordinate variables -----
        
        if len(variable.dimensions) < 3:
            try:
                nc[name].setncatts(f[name].__dict__)  # copy variable attributes all at once via dictionary
                
            except:
                pass
            nc.createVariable(name, variable.datatype, variable.dimensions)
            nc[name][:] = f[name][:]
                
        else:
            for bc_name in ["_W_bry", "_E_bry"]:
                print('@1: name + bc_name =', name + bc_name)                
                nc.createVariable(name + bc_name, variable.datatype, variable.dimensions[:-1])
                try:
                    nc[name + bc_name].setncatts(f[name].__dict__)
                except:
                    pass
                slc = [slice(None)] * (len(variable.dimensions) - 1)  

                slc += [0] if bc_name == "_W_bry" else [-1]
                nc[name + bc_name][:] = f[name][slc]  # [slc] is equiv. to [:, :, .., 0] or [:, :, .., -1]

            for bc_name in ["_S_bry", "_N_bry"]:
                print('@2: name + bc_name =', name + bc_name)
                nc.createVariable(name + bc_name, variable.datatype,
                                  variable.dimensions[:-2] + variable.dimensions[-1:])
                try:
                    nc[name + bc_name].setncatts(f[name].__dict__)
                except:
                    pass
                slc = [slice(None)] * (len(variable.dimensions) - 2)

                slc += [0] if bc_name == "_S_bry" else [-1]
                slc += [slice(None)]
                nc[name + bc_name][:] = f[name][slc]
                    
    nc.close()
    f.close()


if __name__ == '__main__':
    main()
