# -*- coding: utf-8 -*-
"""

@author: LCR

"""

from netCDF4 import Dataset

import os



result = 'longitude latitude VSDX file\n'
for root,dirs,files in os.walk('VSDX_WesternPacific'):
    files.sort()
    for f in files:
        print(f)
        fdir = os.path.join(root,f)
        
        ncdata = Dataset(fdir)

        lon = ncdata.variables['longitude'][:]
        lat = ncdata.variables['latitude'][:]
        
        vsdx = ncdata.variables['VSDX'][:]

        vsdx_ = [v[0][0] for v in vsdx]

        vsdx_mean = sum(vsdx)/len(vsdx_)
        result += "%.2f %.2f %f %s\n"%(lon[0],lat[0],vsdx_mean[0][0],f)


with open('VSDX_WesternPacific.txt', 'w') as f:
    f.write(result)

