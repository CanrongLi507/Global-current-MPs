# -*- coding: utf-8 -*-
"""

@author: LCR

"""

from netCDF4 import Dataset

import os



result = 'longitude latitude VHM0 file\n'
for root,dirs,files in os.walk('VHM0_WesternPacific'):
    files.sort()
    for f in files:
        print(f)
        fdir = os.path.join(root,f)
        
        ncdata = Dataset(fdir)

        lon = ncdata.variables['longitude'][:]
        lat = ncdata.variables['latitude'][:]
        
        vhm0 = ncdata.variables['VHM0'][:]

        vhm0_ = [v[0][0] for v in vhm0]

        vhm0_mean = sum(vhm0)/len(vhm0_)
        result += "%.2f %.2f %f %s\n"%(lon[0],lat[0],vhm0_mean[0][0],f)


with open('VHM0_WesternPacific.txt', 'w') as f:
    f.write(result)

