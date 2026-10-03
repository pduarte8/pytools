1) Directory contents

The directory CICE_boundaries contains scripts to create CICE boundaries for the S4K model from different sources as detailed below. These scripts may be used for any other model as long as template files are replaced to define the grid structure and size.

i) make_CICE_bc_from_TOPAZ_ver0.7_trimming_ITD_1.py and make_CICE_bc_from_TOPAZ_ver0.7_trimming_ITD_2.py were produced by Hiroshi Sumata to produce boundaries from TOPAZ sea ice output. the sufixes ITD_1 and ITD_2 correspond to two different sea ice size distributions represented statistically in each macro. To use these files some editing may necessary, regarding file paths and the year for which boundaries are to be produced. This is described in detail in the following paper:
Sumata, H., Granskog, M.A., Duarte, P., 2026. A suite of coupled ocean-sea ice simulations examining the effect of regime shift in sea-ice thickness distribution on ice-ocean interaction in the Arctic Ocean. Geoscientific Model Development 19(2): 647-659, https://doi.org/10.5194/gmd-19-647-2026.

ii) make_CICE_bc_from_NORESM_ver2.0_trimming.py is to produce boundaries from NORESM ice output. This script was developed with the help of ChatUiT (thread "Review NORESM Sea Ice Src") from previous versions now stored in the SandBox. Before the script was inplemented, several other scripts were used to double-check NORESM variable meaning and determine the best way to convert them to CICE variables and other details. These have the prefix Check*.

2) NORESM details

•	NorESM typically uses 5 ice thickness categories by default, following the standard CICE framework.
•	The thickness categories have the following maximums in m: 0.6445072, 1.391433, 2.470179, 4.567288 and 1e+08. These may double-checked by reading the variables NCAT in the NORESM output files. These categories correspond to kcatbound = 0 in ice_in file, which is the same setting we use in S4K. I double-checked and I am using exactly the same NCAT in S4K_bgc.
•	NORESM uses 3 layers of snow and 8 layers of ice
•	It does not have sea ice salinity data. Therefore, I will have to use the same approach I used before to compute salinity based on assumed profiles for FYI and for MYI and selecting the former or the latter depending on ice age (provided in seconds by NORESM). 
•	 Sea ice temperature profiles have to be calculated as before, assuming a linear change between the top and the bottom of the ice with the following endmembers: sitempsnice (snopw-ice interface temperature) or sitemptop (sea ice surface temperature in the absence of snow) and sitempbot (sea ice bottom temperature). Temperatures are in K.
•	Snow temperature is averaged between sitemptop (I am assuming that this stands for temperature at the top of the snow or the ice (see above), when snow is absent. 

________________________________________

2.1) NORESM category variables and target CICE variables
   
The relevant NORESM fields are:
aicen          category ice area fraction
siitdthick     category local ice thickness
siitdsnthick   category local snow thickness
hi             grid-cell mean ice thickness
hs             grid-cell mean snow thickness
The following source-grid consistency checks were performed.

Ice

The relation:

hi = Σ [aicen_n × siitdthick_n]

was verified to floating-point precision. Therefore, the target CICE category ice-volume variable is constructed on the NORESM grid as:

vicen_n = aicen_n × siitdthick_n

Snow

The relation:

hs = Σ [aicen_n × siitdsnthick_n]

was also verified to floating-point precision. Therefore, the target CICE category snow-volume variable is:

vsnon_n = aicen_n × siitdsnthick_n

The diagnostic variable:
snowfracn
is not used in this conversion. Multiplying snow thickness by snowfracn produced a mismatch with NORESM hs.

Final category mapping

NORESM quantity	Target CICE variable	Construction
aicen[n]	aicen[n]	Directly regridded
aicen[n] * siitdthick[n]	vicen[n]	Construct before regridding
aicen[n] * siitdsnthick[n]	vsnon[n]	Construct before regridding

________________________________________

2.2) Horizontal regridding
   
Regridding method
The script currently uses:
CDO remapbil
for all fields.
This includes:
aicen
vicen
vsnon
siage
ardg
sitemptop
sitempsnic
sitempbot
uvel/vvel-related earth-relative velocity components
Why remapbil is used
Conservative remapping (remapcon) was investigated because it would ideally preserve ice area, volume, and snow volume.
NORESM provides source-cell bounds:
lont_bounds
latt_bounds
lonu_bounds
latu_bounds
However, the target regional CICE reference files contain only cell-centre coordinates:
TLON, TLAT
ULON,
The target CICE B-grid geometry was investigated. It was confirmed that, in the interior:
T[j, i] has U-grid corners:

U[j-1, i-1]
U[j-1, i]
U[j,   i]
U[j,
with sub-metre centre/vertex-centroid agreement.
However, the cropped target grid lacks the western and southern U-grid halo points required to define the polygons of the outermost target T cells. These are exactly cells that matter for boundary extraction.
Therefore, target polygons cannot be built safely for all cells, and conservative remapping should not be used unless the original untrimmed target grid becomes available.
remapbil is therefore the appropriate practical choice.
This does not mean that the interpolation ignores geographic coordinates. CDO uses source and target longitude/latitude to locate points correctly on the curvilinear geographical grids. The limitation is that remapbil is not area/volume conservative.

________________________________________

2.3) Target mask and treatment of invalid cells
   
The target mask file:
cice.kmt.nc
contains:
kmt(454, 696)
``
with:
kmt = 1  ocean
It was compared with tmask from the target reference file:
Wet cells in kmt:     273688
Wet cells in tmask:   273688
Mask differences:          0
Thus:
target_o
is used as the target land/ocean mask.
Missing-value policy
The final output must contain no NaN values because CICE/ROMS input handling can be unreliable when NaNs are present.
The script uses:
FILL_VALUE = -9999.0
The policy is:

________________________________________

2.4) Temporal interpolation: monthly NORESM to daily boundary forcing
   
NORESM source data are monthly mean fields, but the boundary forcing is required daily.
The source months used are:
December of the previous year
January to December
For example, for 2023:
2022-12
2023-01
...
2023-12
Each monthly mean is assigned to the midpoint of its month. The script then linearly interpolates all regridded source state fields to daily output times.
This use of the preceding December and following January avoids temporal extrapolation at the beginning and end of the target year.

________________________________________

2.5) Category-state cleanup after interpolation
   
After horizontal and temporal interpolation, the script applies physical constraints:
aicen = np.maximum(aicen, 0.0)
vicen = np.maximum(vicen, 0.0)
vsnon = np.maximum(vsnon, 0.0)
The total category concentration is:
aice = Σ aicen_n

If numerical interpolation causes:
aice > 1 
the script rescales all category state variables consistently:
aicen
This preserves category-local ice thickness:
hice_n = vicen_n / aicen_n
and category-local snow depth:
hsnow_n = vsnon_n / aicen_n

This is important. An earlier script scaled only aicen, which would have changed category thicknesses artificially.
The monthly test produced:
Maximum sum(aicen): 0.999991, which is physically consistent.

________________________________________

2.6) Sea-ice age and salinity profiles
   
NORESM provides:
siage
units = s
The target CICE iage variable is written in days:
iage_days = siage_seconds / 86400
For salinity, age is converted to model years:
age_years = siage_seconds / (365 * 86400)
The FYI/MYI classification is:
FYI: siage <= 1 year
MYI: siage > 1 year
The original FYI salinity profile is retained for FYI, while the original MYI profile is retained for MYI.
The old script classified FYI/MYI using a thickness threshold of 1.5 m. This was replaced by ice age, which is more physically meaningful.
siage has no category dimension, so all categories in one grid cell receive the same FYI/MYI classification.

________________________________________

2.7) Ice and snow temperatures
   
The script uses direct NORESM CICE temperature diagnostics, rather than atmospheric temperatures:
Physical interpretation
For snow-covered ice:
atmosphere
    |
sitemptop       snow surface temperature
    |
snow layer
    |
sitempsnic      snow–ice interface / ice-top temperature
    |
ice layers
    |
sitempbot       ice-bottom temperature
    |
ocean

For snow-free ice:
atmosphere
    |
sitemptop       ice-top temperature
    |
ice layers
    |
sitempbot       ice-bottom temperature
    |
ocean

Category-specific ice-top temperature

The script calculates the category-local snow depth as:
hsnow_n = vsnon_n / aicen_n
where aicen_n is greater than zero.
If snow is present in a category:
ice
That is, the snow–ice interface temperature is used as the upper boundary condition for the ice-temperature profile.
If snow is absent in a category:
ice_top_temp = sitemptop
That is, the sea-ice surface temperature is used as the upper boundary condition for the ice-temperature profile.
In Python, this is expressed as:
ice_top_temp
With one snow layer, the snow-layer temperature is set to the midpoint of the temperature profile between the snow surface and the snow–ice interface:
Tsnow = (sitemptop + sitempsnic) / 2

The target CICE ice-temperature profile contains seven layers. The temperature in each ice layer is obtained by linear interpolation between:
ice_top_temp    
For ice layer k, where k = 0, 1, ..., 6:
z_k = (k + 0.5) / 7
Tice_k = ice_top_temp + z_k × (sitempbot - ice_top_temp)

________________________________________

2.8) Ridged and level ice

NORESM provides:
ardg
long_name = ridged ice area fraction
units = 
The following diagnostic was performed:
Maximum aice:             0.9999983
Maximum ardg:             0.9231697
Maximum ardg - aice:      0.0
Cells with ardg > aice:  
Thus:
ardg >= 0
and ardg is interpreted as the ridged-ice area fraction per total grid-cell area.
The total level-ice area is:
alevel = aice - ardg
Since ardg has no category dimension, ridged and level ice are partitioned proportionally over the categories.
The level-ice factor is:
level_factor = (aice - ardg) / aice
bounded between zero and one.
0 <= level_factor <= 1
Then:
alvln_n = aicen_n × level_factor
vlvln_n = vicen_n × level
This guarantees:
Σ alvln_n = aice - ardg 
and preserves the category-local thickness of level ice.
This is more physically consistent than the old formula because ardg is an absolute grid-cell area fraction, not a fraction of every category individually.

________________________________________

2.9) Velocity processing
    
NORESM provides grid-relative velocity components:
siu
siv
ANGLE
The procedure is:
NORESM siu/siv in source CICE grid coordinates
    ↓
rotate with source ANGLE
    ↓
eastward/northward velocity
    ↓
bilinear interpolation to target U grid
    ↓
rotate with target ANGLE
    ↓
target CICE uvel/vvel
The source-to-earth rotation is:
u_east = (
    siu * np.cos(source_angle)
    - siv * np.sin(source_angle)
)

v_north = (
    siu * np.sin(source_angle)
    + siv * np.cos(source_angle)
)
The earth-to-target rotation is:
uvel = (
    u_east_target * np.cos(target_angle)
    + v_north_target * np.sin(target_angle)
)

vvel = (
    -u_east_target * np.sin(target_angle)
   
The script removes invalid or unrealistic target velocities using:
MAX_ICE_SPEED = 10.0  # m s-1
and replaces such values with zero over valid ocean cells.
________________________________________

2.10) Other boundary fields
    
The script creates all main target CICE boundary variables:
aicen
vicen
vsnon
alvln
vlvln
Tinz
Sinz
Tsnz
Tsfc
iage
uvel
vvel
ap
The following currently have no direct NORESM mapping in the script and are initialized to zero over ocean cells:
apondn
hpondn
ipond
They receive -9999.0 over land.
________________________________________

2.11) Full-domain and trimmed files
    
The script first creates a temporary full-domain file:
cice_bc_from_NORES
This is useful for diagnostics and visual inspection but can be large for daily output.
It then extracts the boundary slices and creates:
cice_bc_from_NORESM.trimmed.YYYY.nc
``
with variables such as:
aicen_W_bry
aicen_E_bry
aicen_S_bry
aicen_N_bry

vicen_W_bry
vicen_E_bry
vicen_S_bry
vicen_N_bry

uvel_W_bry
uvel_E_bry
uvel_S_bry
uvel_N_bry
The trimmed file is the intended production boundary file.
For production use, it is reasonable to delete the full-domain file after successful trimming:
KEEP_FULL_DOMAIN_FILE = False

if not KEEP_FULL_DOMAIN_FILE:
   
________________________________________

2.12) Precision and file size
    
Daily full-domain thermodynamic fields are large, particularly:
Tinz
Sinz
Use NetCDF float32 for output data variables:
"f4"
rather than float64:
"f8"
Time may remain float64:
time
For the full-domain temporary NetCDF4 file, compression can also be enabled:
z
The trimmed boundary file is much smaller because it contains only the four domain edges.
________________________________________

2.13) Recommended testing sequence
    
Short test
Use:
YEAR = 2023

TEST_START = "202
This writes 31 daily records.
Check:
TIME = 31
Verify no NaNs in all output variables.
Check:
0 <= sum(aicen) <=
Inspect maps of:
sum(aicen)
sum(vicen)
sum(vsnon)
ardg
al
and inspect the four extracted boundaries.
Full-year production run
After validation:
TEST_START = None
TEST_END = None
This writes 365 daily records for a normal year.
Keep the final trimmed file:
cice_bc_from_NORESM.trimmed.YYYY.nc
and, if storage is limited, remove the full-domain intermediate file after trimming.
