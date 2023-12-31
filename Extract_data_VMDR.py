# -*- coding: utf-8 -*-
"""

@author: LCR

"""

from netCDF4 import Dataset

import os



result = 'longitude latitude VMDR file\n'
for root,dirs,files in os.walk('VMDR_WesternPacific'):
    files.sort()
    for f in files:
        print(f)
        fdir = os.path.join(root,f)
        
        ncdata = Dataset(fdir)

        lon = ncdata.variables['longitude'][:]
        lat = ncdata.variables['latitude'][:]
        
        vmdr = ncdata.variables['VMDR'][:]

        vmdr_ = [v[0][0] for v in vmdr]

        vmdr_mean = sum(vmdr)/len(vmdr_)
        result += "%.2f %.2f %f %s\n"%(lon[0],lat[0],vmdr_mean[0][0],f)


with open('VMDR_WesternPacific.txt', 'w') as f:
    f.write(result)

