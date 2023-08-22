# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:25:17 2023

@author: hybth
"""
import numpy as np
import scipy.stats as st

# SPEI线性变化趋势空间分布图
def cal_linreg_trend(data, dem, p=0.05):
    '''
    Parameters
    ----------
    data : 3d narray, time * lat * lon
    dem  : 2d narray, lat * lon
    p    : float
    Returns
    -------
    lin_trend: 2d narray, lat * lon
    sign_result: 2d narray, lat * lon, -2 显著下降；-1 不显著下降； 0 不变； 1 不显著增加；2 显著增加
    '''
    times      = np.size(data, axis=0)
    rows       = np.size(data, axis=1)
    cols       = np.size(data, axis=2)
    lin_trend  = np.full((rows,cols), np.nan)
    sig_result = np.full((rows,cols), np.nan)
    
    for row in range(rows):
        for col in range(cols):
            if ~np.isnan(dem[row, col]):
                data_series  = data[:,row, col]
                slope, intercept, r_value, p_value, std_err = st.linregress(range(times), data_series)
                lin_trend[row, col]  = slope
                
                if slope<0 and p_value<p:
                    sig_result[row, col] = 2
                elif slope<0 and p_value>p:
                    sig_result[row, col] = 1
                elif slope>0 and p_value>p:
                    sig_result[row, col] = 1
                elif slope>0 and p_value<p:
                    sig_result[row, col] = 2
                else:
                    sig_result[row, col] = np.nan

    return lin_trend, sig_result

from scipy import stats
from matplotlib import pyplot as plt 
 
def sk(data):
    n=len(data)
    Sk     = [0]
    UFk    = [0]
    s      =  0
    E      = [0]
    Var    = [0]
    for i in range(1,n):
        for j in range(i):
            if data[i] > data[j]:
                s = s+1
            else:
                s = s+0
        Sk.append(s)
        E.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UFk.append((Sk[i]-E[i])/np.sqrt(Var[i]))
    UFk=np.array(UFk)
    return UFk

#a为置信度
def MK(data, a):
    ufk=sk(data)          #顺序列
    ubk1=sk(data[::-1])   #逆序列
    ubk=-ubk1[::-1]        #逆转逆序列
    
    #输出突变点的位置
    p=[]
    u=ufk-ubk
    for i in range(1,len(ufk)):
        if u[i-1]*u[i]<0:
            p.append(i)                
    return ufk, ubk, p


from eofs.standard import Eof
def eof_analys(data,lat):
    #计算权重：纬度cos值开方
    coslat = np.cos(np.deg2rad(lat))
    wgts = np.sqrt(coslat)[..., np.newaxis] 
    
    #做EOF分析
    solver = Eof(data, weights = wgts)
    EOFs   = solver.eofsAsCorrelation()#空间模态
    PCs    = solver.pcs(npcs   = 3, pcscaling = 1)#时间序列，取前三个模态
	
	#方差贡献
    eigen_Values   = solver.eigenvalues()
    percentContrib = eigen_Values * 100./np.sum(eigen_Values)
	
	#返回空间模态，时间序列和方差贡献
    return EOFs,PCs,percentContrib
