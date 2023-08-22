#1.进行环境和路径设置

import matplotlib.pyplot as plt
from climate_indices import indices, compute
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from datetime import timedelta
import netCDF4 as nc
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import os             
import sys
import cmaps
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import scipy.stats as st

# 用来正常显示中文标签、负号
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# 设置gdal的投影，可以在相应位置找到这个文件
os.environ['PROJ_LIB'] = r'C:\Users\haojiang66\anaconda3\envs\testev\Lib\site-packages\osgeo\data\proj'  

# 设置一些路径，在每个代码块都会有某些路径被用到

lib_path = os.path.join(os.path.dirname(__file__), 'functions')  
#将当前工作目录下的 lib_path 路径添加到 Python 解释器
sys.path.append(os.path.abspath(lib_path))

data_root = os.path.join(os.path.dirname(__file__), 'data_yours')
# China SHP file path  
china_shp_path = os.path.join(data_root, 'shp', 'china', 'china.shp')
# DEM file path  
dem_path = os.path.join(data_root, 'dem', 'dem_010.tif')
# SHP file path  
shp_file_path = os.path.join(data_root, 'shp', '长江流域面.shp')  

print('path and envs set over')

#9.计算spei
###########################################################################################################################################################
t2m_file_path = data_root + r'\t2m_warp.nc'
tp_file_path = data_root + r'\tp_warp.nc'

f = nc.Dataset(t2m_file_path)
lon = np.array(f.variables['lon'][:])
lat = np.array(f.variables['lat'][:])
time = np.array(f.variables['time'][:])
bandCount = np.size(time, axis=0)
temp = np.array(f.variables['temp'])
prec = np.array(nc.Dataset(tp_file_path).variables['prec'])

dem = gdal.Open(dem_path)
dem = dem.GetRasterBand(1).ReadAsArray()

#####
#              time可能需要修改
######
time = pd.date_range('19500101', '20201201', freq='MS').values.astype('datetime64[s]').tolist()
print('data load over')

spei_scale = 12  # 计算12个月尺度的SPEI，反映过去一年内的水分亏损情况
spei = np.full((len(time), len(lat), len(lon)), np.nan)
# for row in tqdm(range(len(lat))):
#%%
for row in range(len(lat)):
    for col in range(len(lon)):
        if dem[row, col] > 0:
            try:
                slat = lat[row]
                temp_monthly = temp[:, row, col]
                prec_monthly = prec[:, row, col]
                # 使用TW法计算潜在蒸散发，输入温度、纬度即可
                pet_monthly = indices.pet(temp_monthly, slat, data_start_year=1950)

                spei[:, row, col] = indices.spei(prec_monthly,
                                                 pet_monthly,
                                                 scale=spei_scale,
                                                 distribution=indices.Distribution.gamma,
                                                 periodicity=compute.Periodicity.monthly,
                                                 data_start_year=1950,
                                                 calibration_year_initial=1950,
                                                 calibration_year_final=2020)
            except Exception as r:
                print(row, col)

from functions import create_nc

output_file_path = data_root + r'\spei_' + str(spei_scale) + '.nc'
create_nc(output_file_path, spei, "SPEI", lon, lat, time)
print('nc out over')

#10.计算年尺度spei
###########################################################################################################################################################
# %% step1 导入数据
spei_path = data_root + r'\spei_12.nc'
f = nc.Dataset(spei_path)
spei = np.array(f.variables['SPEI'][:])
lon = np.array(f.variables["lon"][:])
lat = np.array(f.variables["lat"][:])

g = gdal.Open(dem_path)
dem = g.GetRasterBand(1).ReadAsArray()
A = dem >= 0
NumRaster = np.sum((A))
print('研究区内栅格数：{}'.format(NumRaster))

# %% step2 处理SPEI中的缺失值，一维线性插值
spei = spei[12:]  # 因为程序计算1950年前12个数为nan，插值出来的结果不可靠，所以1950年没包括在分析当中。
print('------------------------------------------------------')
spei_value = spei[:, 40,250]  
print(spei_value)
print('------------------------------------------------------')

A = np.tile(A, (len(spei), 1, 1))
spei_new = np.copy(spei[A]).reshape(len(spei), NumRaster)  # 月数*栅格数
for i in range(NumRaster):
    y = spei_new[:, i]
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    if not np.all(np.isnan(y)):
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        spei_new[:, i] = y
for i in range(len(spei)):
    spei[i][A[i]] = spei_new[i]
print('------------------------------------------------------')
print(spei.shape)
print(np.nanmax(spei[:,:]))
print('------------------------------------------------------')
print(lon)
print(lat)
print('------------------------------------------------------')

# %% step3 统计历年干旱情况，选择每年12月的SPEI作为这一年干旱的表征指数，具体需要了解spei的计算原理
spei_year = np.array([spei[i * 12, :, :] for i in range(len(spei) // 12)])
spei_year_area = np.nanmean(np.nanmean(spei_year[:], axis=2), axis=1)
data = spei_year_area
fig = plt.figure(figsize=(8, 3), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.bar(np.arange(len(data)), data, color=np.where(data > np.mean(data), 'blue', 'tomato'), alpha=0.5, )
blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='湿润年'.format(np.mean(data)))
tomato_patch = mpatches.Patch(color='tomato', alpha=0.5, label='干旱年'.format(np.mean(data)))
ax.legend(handles=[blue_patch, tomato_patch], ncol=3)
xmajorLocator = MultipleLocator(5)  # 将x主刻度标签设置为a的倍数
xminorLocator = MultipleLocator(1)  # 将x轴次刻度标签设置为n的倍数
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
x_labels = ['{}'.format(i + 1951) for i in np.arange(-1, len(data) + 5, 5)]
ax.set_xticks(np.arange(-1, len(data) + 5, 5), x_labels)
plt.title('年尺度SPEI')
plt.savefig('spei_year.png', dpi=300)


# %% step4 探究SPEI-12年际趋势检验
print('------------------------------------------------------')
max_val_per_axis = np.max(spei_year, axis=1)   
min_val_per_axis = np.min(spei_year, axis=1)  
print("每个维度的最大值：", max_val_per_axis)  
print("每个维度的最小值：", min_val_per_axis)  
print('------------------------------------------------------')
from functions import cal_linreg_trend

lin_trend, sig95 = cal_linreg_trend(spei_year, dem, p=0.05)
print('------------------------------------------------------')
max_val_per_axis = np.nanmax(lin_trend, axis=0)   
min_val_per_axis = np.nanmin(lin_trend, axis=0)  
print("每个维度的最大值：", max_val_per_axis)  
print("每个维度的最小值：", min_val_per_axis)  
print('------------------------------------------------------')
fig = plt.figure(figsize=(6, 6), dpi=300)
ax = fig.add_subplot(1, 1, 1)
plt.pcolormesh(lon, lat, lin_trend, vmin=np.nanmin(lin_trend), vmax=np.nanmax(lin_trend), cmap=cmaps.BlueRed_r,shading='auto',linewidths=0.05)
clb = plt.colorbar(orientation="horizontal", pad=0.05)
clb.ax.tick_params(labelsize=10)
plt.contour(lon, lat, sig95, [-99, 2], colors='k', linewidths=0.05)
plt.title('SPEI线性变化趋势和95%显著性空间分布图')
plt.savefig('spei_analysis.png', dpi=300)


# %% step5 探究SPEI-12年际突变检测
from functions import MK

ufk, ubk, tipping = MK(spei_year_area, 0.95)

data = spei_year_area
from scipy.stats import norm

conf_intveral = norm.interval(0.95, loc=0, scale=1)  # 获取置信区间
fig = plt.figure(figsize=(8, 3), dpi=300)
ax1 = fig.add_subplot(1, 1, 1)
ax1.bar(np.arange(len(data)), data, color=np.where(data > np.mean(data), 'blue', 'tomato'), alpha=0.5, )

ax1.set_ylabel('SPEI', fontsize=20)
ax1.set_ylim(-2, 2)
ax2 = ax1.twinx()
l1, = ax2.plot(range(len(data)), ufk, label='UFk', color='b', alpha=0.8)
l2, = ax2.plot(range(len(data)), ubk, label='UBk', color='r', alpha=0.8)
ax2.set_ylabel('UFk-UBk', fontsize=20)
x_lim = plt.xlim()
ax2.set_ylim([-6, 6])
l3, = ax2.plot(x_lim, [conf_intveral[0], conf_intveral[0]], 'm--', color='r', label='95%显著区间')
ax2.plot(x_lim, [conf_intveral[1], conf_intveral[1]], 'm--', color='r')
ax2.axhline(0, ls="-", c="k", lw=1)

blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='湿润年'.format(np.mean(data)))
tomato_patch = mpatches.Patch(color='tomato', alpha=0.5, label='干旱年'.format(np.mean(data)))
ax1.legend(handles=[blue_patch, tomato_patch, l1, l2, l3], ncol=5)

from matplotlib.ticker import MultipleLocator

xmajorLocator = MultipleLocator(5)  # 将x主刻度标签设置为a的倍数
xminorLocator = MultipleLocator(1)  # 将x轴次刻度标签设置为n的倍数
ax1.xaxis.set_major_locator(xmajorLocator)
ax1.xaxis.set_minor_locator(xminorLocator)
x_labels = ['{}'.format(i + 1951) for i in np.arange(-1, len(data) + 5, 5)]
ax1.set_xticks(np.arange(-1, len(data) + 5, 5), x_labels)
for i in tipping:
    ax2.scatter(i - 1, ubk[i - 1], marker='s', s=20, c='r')
    ax2.axvline(x=i - 1, c='k', ls=':', lw=1)
ax = plt.gca()
plt.text(0.1, 0.15, '突变年份:', fontsize=10, transform=ax.transAxes)
plt.text(0.2, 0.15, ['{}'.format(i + 1951) for i in tipping], fontsize=10, transform=ax.transAxes)
plt.savefig('mk.png', dpi=300)


# %% step6 小波分析，寻找各个感兴趣区域的干旱重现期

import sys
import os
lib_path = os.path.join(r'C:\Users\haojiang66\Desktop\mysrt\functions')  
#将当前工作目录下的 lib_path 路径添加到 Python 解释器
sys.path.append(os.path.abspath(lib_path))

from waveletFunctions import wave_signif, wavelet
from plot_functions import draw_wavelet

# monthly area average spei
spei_area = np.nanmean(np.nanmean(spei,axis=2),axis=1)
spei_area = spei_area - np.mean(spei_area)
variance = np.std(spei_area, ddof=1) ** 2
print("variance = ", variance)

if 0:
    variance = 1.0
    spei_area = spei_area / np.std(spei_area, ddof=1)
n = len(spei_area)
dt = 1/12
times = np.arange(len(spei_area)) * dt + 1951  # construct time array
xlim = ([1951, 2020])  # plotting range
pad = 1  # pad the time series with zeroes (recommended)
dj = 0.25  # this will do 4 sub-octaves per octave
s0 = 2 * dt  # this says start at a scale of 6 months
j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
lag1 = 0.72  # lag-1 autocorrelation for red noise background
print("lag1 = ", lag1)
mother = 'MORLET'

# Wavelet transform:
wave, period, scale, coi = wavelet(spei_area, dt, pad, dj, s0, j1, mother)
power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

# Significance levels:
signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
    lag1=lag1, mother=mother)
# expand signif --> (J+1)x(N) array
sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
sig95 = power / sig95  # where ratio > 1, power is significant

# Global wavelet spectrum & significance levels:
dof = n - scale  # the -scale corrects for padding at edges
global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1,
    lag1=lag1, dof=dof, mother=mother)

draw_wavelet(spei_area, period, power, sig95, coi, global_ws, global_signif, times, xlim,)

