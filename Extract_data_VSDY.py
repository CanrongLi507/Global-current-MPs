# -*- coding: utf-8 -*-
"""

@author: LCR

"""

from netCDF4 import Dataset

import os



result = 'longitude latitude VSDY file\n'
for root,dirs,files in os.walk('VSDY_WesternPacific'):
    files.sort()
    for f in files:
        print(f)
        fdir = os.path.join(root,f)
        
        ncdata = Dataset(fdir)

        lon = ncdata.variables['longitude'][:]
        lat = ncdata.variables['latitude'][:]
        
        vsdy = ncdata.variables['VSDY'][:]

        vsdy_ = [v[0][0] for v in vsdy]

        vsdy_mean = sum(vsdy)/len(vsdy_)
        result += "%.2f %.2f %f %s\n"%(lon[0],lat[0],vsdy_mean[0][0],f)


with open('VSDY_WesternPacific.txt', 'w') as f:
    f.write(result)

