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
import os             #os库主要是对文件和文件夹进行操作，在Python中对⽂件和⽂件夹的操作要借助os模块⾥⾯的相关功能。
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
china_shp_path = os.path.join(data_root, 'shp', 'china.shp')
# DEM file path  
dem_path = os.path.join(data_root, 'dem', 'dem_010.tif')
# SHP file path  
shp_file_path = os.path.join(data_root, 'shp', '长江流域面.shp')  

print('path and envs set over')

#2.进行数据的导入
###########################################################################################################################################################
# 导入ERA5-land 0.1°数据
data_file_path =r'C:\Users\haojiang66\Desktop\srt\4\fourth talk\data_yours\cj.nc'

f = nc.Dataset(data_file_path)#利用nc.Dataset(file)读取文件
lon = np.array(f.variables['longitude'][:])
lat = np.array(f.variables['latitude'][:])
time = np.array(f.variables['time'][:])
# ear5的时间是从1900年1月1日0点0分0秒开始的，然后步长以hours累计
time = [datetime.datetime(1900, 1, 1, 0, 0) + datetime.timedelta(hours=int(i)) for i in time]
bandCount = np.size(time, axis=0)
t2m = np.array(f.variables['t2m'])
tp = np.array(f.variables['tp'])
# 判断t2m的维度
if t2m.ndim == 4:
    t2m = t2m[:, 0, :, :] + t2m[:, 1, :, :] + 32767
    tp = tp[:, 0, :, :] + tp[:, 1, :, :] + 32767
print('data load over')


#3.
from functions import nc2tif

output_file_path1 = data_root + r'\t2m.tif'
nc2tif(t2m, lon, lat, time, output_file_path1)
output_file_path2 = data_root + r'\tp.tif'
nc2tif(tp, lon, lat, time, output_file_path2)
print('nc to tif over')

#4.输出resample
from functions import gdal_reproject_image

input_file_path1 = data_root + r'\t2m.tif'
input_file_path2 = data_root + r'\tp.tif'
reference_file_path = dem_path
gdal_reproject_image(input_file_path1, reference_file_path)
gdal_reproject_image(input_file_path2, reference_file_path)
print('resample over')

#5.输出warp
from functions import gdal_warp

input_file_path1 = data_root + r'\t2m_resample.tif'
output_file_path1 = data_root + r'\t2m_warp.tif'
gdal_warp(input_file_path1, shp_file_path, output_file_path1)

input_file_path2 = data_root + r'\tp_resample.tif'
output_file_path2 = data_root + r'\tp_warp.tif'
gdal_warp(input_file_path2, shp_file_path, output_file_path2)
print('warp over')

#6.tif2nc
from functions import tif2nc
input_file_path = data_root + r'\t2m_warp.tif'
tif2nc(input_file_path, out_var_name='temp', time=time, scale=1)
input_file_path = data_root + r'\tp_warp.tif'
tif2nc(input_file_path, out_var_name='prec', time=time, scale=1)
print('tif2nc over')


#7.检查nc文件的详细信息
###########################################################################################################################################################

#8.画图
###########################################################################################################################################################
tif_file_path = data_root + r'\t2m_resample.tif'
raster = gdal.Open(tif_file_path)
data = raster.GetRasterBand(1).ReadAsArray() - 273.15
geoTrans = raster.GetGeoTransform()
rows = raster.RasterYSize
cols = raster.RasterXSize
lon = np.array([geoTrans[0] + i * geoTrans[1] for i in range(cols)])  # 网格左上角坐标
lat = np.array([geoTrans[3] + i * geoTrans[5] for i in range(rows)])
#####
#              time根据实际数据修改，大家正常来说都是1950/1/1开始
######
time = pd.date_range('19500101', '20221201', freq='MS').values.astype('datetime64[s]').tolist()

fig = plt.figure(figsize=(6, 4), dpi=300)
ax1 = fig.add_subplot(1, 1, 1)
pc1 = ax1.pcolormesh(lon, lat, data, vmin=-20, vmax=10, shading='auto', linewidths=0.05)
clb = plt.colorbar(pc1, ax=ax1, orientation="vertical")  # Similar to fig.colorbar(im, cax = cax)
ax1.set_title('{}'.format(time[0].strftime('%Y-%m-%d')))
plt.show()


tif_file_path = data_root + r'\tp_resample.tif'
raster = gdal.Open(tif_file_path)
data = raster.GetRasterBand(1).ReadAsArray() * 1000 * 30  # 单位从m换成月mm
geoTrans = raster.GetGeoTransform()
rows = raster.RasterYSize
cols = raster.RasterXSize
lon = np.array([geoTrans[0] + i * geoTrans[1] for i in range(cols)])  # 网格左上角坐标
lat = np.array([geoTrans[3] + i * geoTrans[5] for i in range(rows)])

fig = plt.figure(figsize=(6, 4), dpi=300)
ax1 = fig.add_subplot(1, 1, 1)
pc1 = ax1.pcolormesh(lon, lat, data, vmin=-20, vmax=10, shading='auto', linewidths=0.05)
clb = plt.colorbar(pc1, ax=ax1, orientation="vertical")  # Similar to fig.colorbar(im, cax = cax)
ax1.set_title('{}'.format(time[0].strftime('%Y-%m-%d')))
plt.show()


tif_file_path = data_root + r'\t2m_warp.tif'
raster = gdal.Open(tif_file_path)
data = raster.GetRasterBand(1).ReadAsArray() - 273.15
geoTrans = raster.GetGeoTransform()
rows = raster.RasterYSize
cols = raster.RasterXSize
lon = np.array([geoTrans[0] + i * geoTrans[1] for i in range(cols)])  # 网格左上角坐标
lat = np.array([geoTrans[3] + i * geoTrans[5] for i in range(rows)])
#####
#              time根据实际数据修改，大家正常来说都是1950/1/1开始
######
time = pd.date_range('19500101', '20221201', freq='MS').values.astype('datetime64[s]').tolist()

fig = plt.figure(figsize=(6, 4), dpi=300)
ax1 = fig.add_subplot(1, 1, 1)
pc1 = ax1.pcolormesh(lon, lat, data, vmin=-20, vmax=10, shading='auto', linewidths=0.05)
clb = plt.colorbar(pc1, ax=ax1, orientation="vertical")  # Similar to fig.colorbar(im, cax = cax)
ax1.set_title('{}'.format(time[0].strftime('%Y-%m-%d')))
plt.savefig('t2m_warp.png', dpi=300)
plt.show()


tif_file_path = data_root + r'\tp_warp.tif'
raster = gdal.Open(tif_file_path)
data = raster.GetRasterBand(1).ReadAsArray() * 1000 * 30  # 单位从m换成月mm
geoTrans = raster.GetGeoTransform()
rows = raster.RasterYSize
cols = raster.RasterXSize
lon = np.array([geoTrans[0] + i * geoTrans[1] for i in range(cols)])  # 网格左上角坐标
lat = np.array([geoTrans[3] + i * geoTrans[5] for i in range(rows)])

fig = plt.figure(figsize=(6, 4), dpi=300)
ax1 = fig.add_subplot(1, 1, 1)
pc1 = ax1.pcolormesh(lon, lat, data, vmin=-20, vmax=10, shading='auto', linewidths=0.05)
clb = plt.colorbar(pc1, ax=ax1, orientation="vertical")  # Similar to fig.colorbar(im, cax = cax)
ax1.set_title('{}'.format(time[0].strftime('%Y-%m-%d')))
plt.savefig('tp_warp.png', dpi=300)
plt.show()


