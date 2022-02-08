"""
========================================================================================================================
Authors: Leonardo Hoinaski - leonardo.hoinaski@ufsc.br                                          Last update: 07/02/2022
         Otávio Nunes - otavio.ufsc93@gmail.com



                                             HospDisaggregation.py


 This is the main script to disaggregate daily hospitalization from DATASUS and
 create netCDF files with aggregated hospitalization in regular areas. 
 
 https://www.worldpop.org/project/categories?id=3
 
 https://www.qualocep.com/

 Inputs:  rootPath: Path to functions
          outPath: Path to outputs folder
          lati: Initial latitude (lower-left)
          latf: Final latitude (upper-right)
          loni: Initial longitude (lower-left)
          lonf: Final longitude (upper-right)
          deltaX: Grid resolution/spacing in x direction
          deltaY: Grig resolution/spacing in y direction
          fileId = identification of your output files
          runOrNotTemporal = Run or not daily temporal profile and daily netCDF
          vulGroup = vunerability group tag.
                 'Total' = Total number of hospitalizations
                 'less14' = Total number of hospitalizations for younger than 14
                 'more60' = Total number of hospitalizations for older than 60
                 'adults' = Total number of hospitalizations for adults (15-59)
                 'mens' = Total number of hospitalizations for mens
                 'womans' = Total number of hospitalizations for womans
                 'blacks' = Total number of hospitalizations for blacks 
                 'whites' = Total number of hospitalizations for whites
                 'brown' = Total number of hospitalizations for browns
                 'indigenous' = Total number of hospitalizations for indigenous
                 'asian' = Total number of hospitalizations for asians
          
 Outputs:
     
     Annual basis netCDF
     'HOSP_annual_'+fileId+'_'+str(deltaX)+'x'+str(deltaY)+'_'+str(hosp.ANO_CMPT[0])+'.nc'
     
     Daily basis netCDF
     'HOSP_daily_'+fileId+'_'+str(deltaX)+'x'+str(deltaY)+'_'+str(hosp.ANO_CMPT[0])+'.nc'
    
     
 External functions: gridding.py, netCDFcreator.py

========================================================================================================================
"""

##----------------------------- Import packages
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from geopandas.tools import sjoin
import xarray as xr
import rioxarray as rio
from shapely.geometry import mapping
from pytictoc import TicToc

t = TicToc()
t.tic()
##----------------------------- Inputs
rootPath = os.getcwd()

# ------ Input path
inpPath = 'internacoes/'
shpPath = 'shapefiles/'

# ------ Hospitalization files
respFile = 'result_resp_2019_final_L.csv'
cardFile = 'result_cardio_2019_final_L.csv'
listCEPfile = 'qualocep_geo.csv'
population = 'bra_ppp_2019_1km_Aggregated_UNadj.tif'

region_of_interest = 'SC_Mun97_region.shp'

#----------------------------- Outputs
# ------ Output paths
outPath='Outputs_nc/'

out_fileResp = 'SC_respiratory' # Code to identify your respiratory output files
out_fileCard = 'SC_Cardio' # Code to identify your cardiorespiratory output files

##------------------------- Setting grid resolution

# Users can change the domain and resolution here.
lati =-30 #lati = int(round(bound.miny)) # Initial latitude (lower-left)

latf = -24 #latf = int(round(bound.maxy)) # Final latitude (upper-right)

loni = -54 #loni = int(round(bound.minx)) # Initial longitude (lower-left)

lonf = -47 #lonf = int(round(bound.maxx)) # Final longitude (upper-right)

deltaX = 0.05 # Grid resolution/spacing in x direction

deltaY = 0.05 # Grig resolution/spacing in y direction

prefix = str(deltaX)+'x'+str(deltaY) # grid definition identification

runOrNotTemporal = 1 # Run or not daily temporal profile and daily netCDF

vulGroup = 'Total'  # ['Total','less14','more60','adults',
                    #'mens','womans','blacks','whites',
                    #'brown','indigenous','asian']

## ---------------------------- Processing
def hospitalization_nc(disease, output_nc):

    # ------ Reading hospitalization data
    hosp = pd.read_csv(os.path.join(inpPath, disease))

    # ------ Reading CEP to latlon file
    listCEP = pd.read_csv(os.path.join(inpPath, listCEPfile), delimiter="|",encoding='utf8')

    # ------ Replacing -  by nan 
    listCEP['longitude']=listCEP['longitude'].replace('-', np.nan)
    listCEP['latitude']=listCEP['latitude'].replace('-', np.nan)

    #---------------------- Hospitalization CEP to latlon

    # ------ Getting latlon of hospitalization data from DATASUS
    lon = dict(zip(listCEP['cep'], listCEP['longitude']))
    lat = dict(zip(listCEP['cep'], listCEP['latitude']))

    hosp['lon'] = hosp['CEP'].map(lon).astype(float)
    hosp['lat'] = hosp['CEP'].map(lat).astype(float)

    # ------ Setting conditions for vulnerability groups
    vulGroups = {'less14': hosp.IDADE < 14,
                 'more60': hosp.IDADE > 60,
                 'adults': ((hosp.IDADE<59) & (hosp.IDADE>15)),
                 'mens': hosp.SEXO == 1,
                 'womans': hosp.SEXO == 3,
                 'blacks': hosp.RACA_COR == 2,
                 'whites': hosp.RACA_COR == 1,
                 'brown': hosp.RACA_COR == 3,
                 'indigenous': hosp.RACA_COR == 5,
                 'asian': hosp.RACA_COR == 4}

    groups = [np.select([vulGroups[groups]], [1], default=0) for groups in list(vulGroups.keys())]
    hosp[list(vulGroups.keys())] = pd.DataFrame(groups).T

    # ------ Varibles 
    variables = list(vulGroups.keys())

    # ------ Converting collumn to datetime
    hosp['DT_INTER']=pd.to_datetime(hosp['DT_INTER'],format='%Y%m%d')
    hosp = hosp.sort_values(by="DT_INTER").reset_index(drop = True)

    # ------ Hospitalization CEP's/city
    aggVars = dict(zip(variables, len(variables)*['sum']))
    aggVars['lon'] = 'first'
    aggVars['lat'] = 'first'

    hosp_ceps = hosp.groupby(['CEP'], as_index = False).agg(aggVars)
    hosp_ceps = hosp_ceps.dropna()
    hosp_ceps = gpd.GeoDataFrame(hosp_ceps, geometry = gpd.points_from_xy(hosp_ceps.lon, hosp_ceps.lat), crs = 'EPSG:4326')
    hosp_ceps['TOTAL'] = hosp_ceps[variables].sum(axis = 1)

    # ------------------------- Creating grid
    ncols = len(np.arange(loni, lonf, deltaX))-1
    nrows = len(np.arange(lati, latf, deltaY))-1

    # --- Grid constructor
    xlon = np.linspace(loni,(loni+ncols*deltaX),ncols)
    ylat = np.linspace(lati,(lati+nrows*deltaY),nrows)

    xv, yv = np.meshgrid(xlon, ylat)

    lon = xv.flatten(); lat = yv.flatten()

    grid = gpd.GeoDataFrame(geometry = gpd.points_from_xy(lon, lat), crs = 'EPSG:4326')
    grid.geometry = grid.buffer(deltaX/2).envelope

    # ------ Hospitalization regrid
    hosp_reg = sjoin(hosp_ceps, grid,  how="left", op="within")

    hosp_reg = hosp_reg.groupby(['index_right'])[variables].sum()

    # ------ Replacing 0 -> np.nan
    hosp_reg = hosp_reg.replace(0, np.nan)

    # ------ Allocate hospitalization data in grid
    grid.loc[hosp_reg.index.astype(int), variables] = hosp_reg
    grid['TOTAL'] = np.nansum(grid[variables], axis = 1)

    # ------------------------- Creating MultiIndex domain
    grid['x'] = grid.geometry.centroid.x
    grid['y'] = grid.geometry.centroid.y
    grid = grid.drop('geometry', 1)

    domain = grid.set_index(["y", "x"], drop = True).to_xarray()

    # ------ Removing nodatavals
    domain = domain.where(domain['TOTAL'] != 0)

    # ------------------------- Aggregating population data in domain
    pop = xr.open_rasterio(os.path.join(inpPath, population))

    # ------ Removing nodatavals
    pop = pop.where(pop.data != pop.nodatavals)

    # ------ Allocating population data in domain pixels
    domain['Population'] = xr.DataArray(pop.sel(y = domain.y, x = domain.x,  method = 'nearest').data[0],
                                       dims=["y", "x"])

    # ------------------ Weighing hospitalizations by the number of inhabitants

    # ------ Select hospitalization data by city
    domain = domain.rio.write_crs("EPSG:4326")
    roi = gpd.read_file(os.path.join(shpPath, region_of_interest))
    domain = domain.rio.clip(roi.geometry.apply(mapping))

    domain['inter_pop'] = domain['TOTAL']/domain['Population']

    domain.to_netcdf(os.path.join(outPath, output_nc+'.nc'))

    return domain

resp = hospitalization_nc(respFile, out_fileResp)
cardio = hospitalization_nc(cardFile, out_fileCard)