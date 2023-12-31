# -*- coding: utf-8 -*-
"""
@author: LCR

"""

from netCDF4 import Dataset

import os



result = 'longitude latitude uo file\n'
for root,dirs,files in os.walk('uo_WesternPacific'):
    files.sort()
    for f in files:
        print(f)
        fdir = os.path.join(root,f)
        
        ncdata = Dataset(fdir)

        lon = ncdata.variables['longitude'][:]
        lat = ncdata.variables['latitude'][:]
        
        uo = ncdata.variables['uo'][:]

        uo_ = [v[0][0] for v in uo]

        uo_mean = sum(uo)/len(uo_)
        result += "%.2f %.2f %f %s\n"%(lon[0],lat[0],uo_mean[0][0],f)


with open('uo_WesternPacific.txt', 'w') as f:
    f.write(result)

