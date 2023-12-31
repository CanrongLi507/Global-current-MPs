# -*- coding: utf-8 -*-
"""

@author: LCR

"""

from netCDF4 import Dataset

import os



result = 'longitude latitude vo file\n'
for root,dirs,files in os.walk('vo_WesternPacific'):
    files.sort()
    for f in files:
        print(f)
        fdir = os.path.join(root,f)
        
        ncdata = Dataset(fdir)

        lon = ncdata.variables['longitude'][:]
        lat = ncdata.variables['latitude'][:]

        vo = ncdata.variables['vo'][:]

        vo_ = [v[0][0] for v in vo]

        vo_mean = sum(vo)/len(vo_)
        result += "%.2f %.2f %f %s\n"%(lon[0],lat[0],vo_mean[0][0],f)


with open('vo_WesternPacific.txt', 'w') as f:
    f.write(result)

