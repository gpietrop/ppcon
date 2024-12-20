import os

from ppcon.generate_profile import generate_profiles_from_input
import netCDF4 as nc

dataset = nc.Dataset(os.getcwd() + '/example_profiles/6901648/MR6901648_109.nc', 'r')
print(dataset.variables.keys())
lat = dataset.variables['LATITUDE'][:]
lon = dataset.variables['LONGITUDE'][:]
temp = dataset.variables['TEMP'][:]
pres_temp = dataset.variables['PRES_TEMP'][:]
psal = dataset.variables['PSAL'][:]
print(psal)
pres_psal = dataset.variables['PRES_PSAL'][:]
doxy = dataset.variables['DOXY'][:]
pres_doxy = dataset.variables['PRES_DOXY'][:]
var = dataset.variables['CHLA'][:]
pres_var = dataset.variables['PRES_CHLA'][:]


generate_profiles_from_input(variable="NITRATE",
                             year=2021,
                             month=12,
                             day=8,
                             lat=lat,
                             lon=lon,
                             tuple_temp=(temp, pres_temp),
                             tuple_psal=(psal, pres_psal),
                             tuple_doxy=(doxy, pres_doxy),
                             tuple_var=(var, pres_var))
