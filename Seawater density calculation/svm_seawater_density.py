# -*- coding: utf-8 -*-
"""
dio:

@author: LCR
"""

import math
import pandas as pd

def getDensity(S, T, P):
    ''' Calculate Pw '''
    a0 = 999.842594
    a1 = 6.793952 * math.pow(10, -2)
    a2 = -9.095290 * math.pow(10, -3)
    a3 = 1.001685 * math.pow(10, -4)
    a4 = -1.120083 * math.pow(10, -6)
    a5 = 6.536332 * math.pow(10, -9)
    Pw = a0 + a1 * T + a2 * T * T + a3 * math.pow(T, 3) + a4 * math.pow(T, 4) + a5 * math.pow(T, 5);

    ''' Calculate Pst0'''
    b0 = 8.24493 * math.pow(10, -1);
    b1 = -4.0899 * math.pow(10, -3);
    b2 = 7.6438 * math.pow(10, -5);
    b3 = -8.2467 * math.pow(10, -7);
    b4 = 5.3875 * math.pow(10, -9);
    c0 = -5.72466 * math.pow(10, -3);
    c1 = 1.0227 * math.pow(10, -4);
    c2 = -1.6546 * math.pow(10, -6);
    d0 = 4.8314 * math.pow(10, -4);
    Pst0 = Pw + (b0 + b1 * T + b2 * math.pow(T, 2) + b3 * math.pow(T, 3) + b4 * math.pow(T, 4)) * S + (
                c0 + c1 * T + c2 * math.pow(T, 2)) * S * math.sqrt(S) + d0 * S * S;

    ''' Calculate K(s,t,p) K(s,t,p)=K(s,t,0) +A*P +B*P*P    Set as Kstp'''
    h0 = 3.239908;
    h1 = 1.43713 * math.pow(10, -3);
    h2 = 1.16092 * math.pow(10, -4);
    h3 = -5.77905 * math.pow(10, -7);
    Aw = h0 + h1 * T + h2 * T * T + h3 * T * T * T;
    '''Bw'''
    k0 = 8.50935 * math.pow(10, -5);
    k1 = -6.12293 * math.pow(10, -6);
    k2 = 5.2787 * math.pow(10, -8);
    Bw = k0 + k1 * T + k2 * T * T;
    '''Kw'''
    e0 = 19652.21;
    e1 = 148.4206;
    e2 = -2.327105;
    e3 = 1.360477 * math.pow(10, -2);
    e4 = -5.155288 * math.pow(10, -5);
    Kw = e0 + e1 * T + e2 * T * T + e3 * math.pow(T, 3) + e4 * math.pow(T, 4);
    '''B'''
    m0 = -9.9348 * math.pow(10, -7);
    m1 = 2.0816 * math.pow(10, -8);
    m2 = 9.1697 * math.pow(10, -10);
    B = Bw + (m0 + m1 * T + m2 * T * T) * S;
    '''A'''
    i0 = 2.2838 * math.pow(10, -3);
    i1 = -1.0981 * math.pow(10, -5);
    i2 = -1.6078 * math.pow(10, -6);
    j0 = 1.91075 * math.pow(10, -4);
    A = Aw + (i0 + i1 * T + i2 * T * T) * S + j0 * S * math.sqrt(S);
    '''K(s,t,0)'''
    f0 = 54.6746;
    f1 = -0.603459;
    f2 = 1.09987 * math.pow(10, -2);
    f3 = -6.1670 * math.pow(10, -5);
    g0 = 7.944 * math.pow(10, -2);
    g1 = 1.6483 * math.pow(10, -2);
    g2 = -5.3009 * math.pow(10, -4);
    Kst0 = Kw + (f0 + f1 * T + f2 * T * T + f3 * math.pow(T, 3)) * S + (g0 + g1 * T + g2 * T * T) * S * math.sqrt(S);
    '''Kstp'''
    Kstp = Kst0 + A * P + B * P * P;

    '''Pstp //p(s,t,p)=p(s,t,0)/[1-p/k(s,t,p)]'''
    Pstp = Pst0 / (1 - (P / Kstp));

    return Pstp

# Read in an Excel spreadsheet #
# p needs to convert the pressure unit first, and then update it into the table #
data = pd.read_excel('svm_seawater_density.xlsx',header=0)
# Defines a list to hold the results of density calculations #
density = []
# Loop through the table #
for i in data.index:
    # Calculated density #
    density_ = getDensity(data['SALT'][i], data['THETA'][i], data['Pressure[db]'][i])
    # Density calculation results are added to the list #
    density.append(density_)
# Assigns a list of density calculations to the table #
data['Density'] = density
# Export to an Excel spreadsheet #
data.to_excel('svm_seawater_density_new.xlsx')

#End#