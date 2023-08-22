# Author: huang yangbin
# CreatTime: 2022/11/23
# FileName: plot_functions
# Description:
from shapely import speedups
speedups.disable()
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime
from datetime import timedelta
import netCDF4 as nc
import os
os.environ['PROJ_LIB'] = r'D:\Anaconda\Lib\site-packages\osgeo\data\proj'



def mapart(ax, root_path):
    '''
    添加地图元素
    '''
    proj=ccrs.PlateCarree()
    ax.coastlines(color='k',lw=0.5)
    # ax.add_feature(cfeature.LAND, facecolor='white')
    #设置地图范围
    ax.set_extent([100, 130, 35, 50],crs=ccrs.PlateCarree())
    #设置经纬度标签
    # ax.set_xticks([100,110,120,130], crs=proj)
    # ax.set_yticks([35,40,45,50], crs=proj)
    
    # 导入研究区边界文件
    shp_path = root_path + '\\Data\\shp\\result.shp'
    shp_map = cfeat.ShapelyFeature(shpreader.Reader(shp_path).geometries(), proj, edgecolor='k', facecolor='none')
    ax.add_feature(shp_map, linewidth=1)
    
    # 导入中国省界边界文件
    china_shp = root_path + '\\Data\\shp\\bou2_4l.shp'
    china_map = cfeat.ShapelyFeature(shpreader.Reader(china_shp).geometries(), proj, edgecolor='k', facecolor='w')
    ax.add_feature(china_map, linewidth=0.2, edgecolor='black', facecolor='w')
    # ax.add_feature(cfeat.LAND.with_scale('10m'))
    
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # lat_formatter = LatitudeFormatter()
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)
    # 设置横纵轴标签格式
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1.2, color='k', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶端的经纬度标签
    gl.right_labels = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(95, 145 + 5, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(-5, 90, 5))
  

def plot_area_ax(data, lons, lats, vmin, vmax, title, root_path, cmap=None):
    proj = ccrs.PlateCarree()  # 创建投影，选择cartopy的默认投影
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_extent([98, 132, 35, 52], crs=ccrs.PlateCarree())
    # 绘制基础信息
    mapart(ax, root_path)    
    # 绘制图像
    plt.pcolormesh(lons, lats, data, transform=proj, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto', linewidths=0.05)    
    # 绘制colorbar
    clb = plt.colorbar(orientation="horizontal", pad=0.05)
    clb.ax.tick_params(labelsize=10)
    # clb.ax.set_xlabel(title, fontsize=14)

    plt.title(title)
    plt.show()
    plt.close()

    
def plot_area_sig(data, signif, lons, lats, vmin, vmax, title, root_path, cmap=None):
    proj = ccrs.PlateCarree()  # 创建投影，选择cartopy的默认投影
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_extent([98, 132, 35, 52], crs=ccrs.PlateCarree())
    # 绘制基础信息
    mapart(ax, root_path)    
    # 绘制图像
    plt.pcolormesh(lons, lats, data, transform=proj, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto', linewidths=0.05)    
    # 绘制colorbar
    clb = plt.colorbar(orientation="horizontal", pad=0.05)
    clb.ax.tick_params(labelsize=10)
    # clb.ax.set_xlabel(title, fontsize=14)
    # 绘制等高线
    plt.contour(lons, lats, signif, [-99, 1], colors='k', linewidths=0.5)
    # cone-of-influence, anything "below" is dubious
    
    # plt.tight_layout()
    # plt.savefig('LASSO_coef')
    plt.title(title)
    plt.show()
    plt.close()
    return


def MK_test(data,ufk,ubk,tipping):
    from scipy import stats
    conf_intveral = stats.norm.interval(0.95, loc=0, scale=1)   #获取置信区间   
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
    b=ax1.bar(range(len(data)),data, color='b', alpha=0.5, label='SPEI')
    #对y值大于0设置为蓝色  小于0的柱设置为绿色
    for bar,height in zip(b,data):
        if height<0:
            bar.set(color='orange', alpha=0.5)
    ax1.set_ylabel('SPEI',fontsize=20)
    ax1.set_ylim(-2,2)
    ax1.legend(loc='lower left',frameon=False,ncol=3,fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(range(len(data)),ufk,label = 'UFk',color = 'b', alpha=0.8)
    ax2.plot(range(len(data)),ubk,label = 'UBk',color = 'r', alpha=0.8)
    ax2.set_ylabel('UFk-UBk',fontsize=20)
    x_lim = plt.xlim()
    ax2.set_ylim([-6,7])
    ax2.plot(x_lim,[conf_intveral[0],conf_intveral[0]],'m--',color='r',label='95%显著区间')
    ax2.plot(x_lim,[conf_intveral[1],conf_intveral[1]],'m--',color='r')
    ax2.axhline(0,ls="--",c="k")
    ax2.legend(loc='upper right',frameon=False,ncol=3,fontsize=16) # 图例
    
    x_labels=['{}'.format(i+1961) for i in np.arange(0,len(data)+5,5)]
    ax1.set_xticks(np.arange(0,len(data)+5,5),x_labels)
    from matplotlib.ticker import MultipleLocator
    xmajorLocator   = MultipleLocator(5) #将x主刻度标签设置为a的倍数
    xminorLocator   = MultipleLocator(1) #将x轴次刻度标签设置为n的倍数
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax = plt.gca()
    if tipping:
        print("突变点年份：")
        print(['{}'.format(i+1961) for i in tipping])
    else:
        print("未检测到突变点")
    plt.text(0.1,0.15,'突变年份', fontsize=12, transform = ax.transAxes)
    plt.text(0.3,0.15,['{}'.format(i+1961) for i in tipping], fontsize=12, transform = ax.transAxes)
    plt.show()
    return


def eof_contourf(EOFs,PCs,pers,A,lats,lons,cmap,root_path,title):
    plt.close
    fig = plt.figure(figsize=(8,8),dpi=300)
    proj=ccrs.PlateCarree()
    year=range(len(A))
    ax1 = fig.add_subplot(3,2, 1, projection=proj)
    mapart(ax1, root_path)
    EOF1=np.full((111,209),np.nan)
    EOF1[A[0]] = EOFs[0]
    p = ax1.contourf(lons,lats,EOF1 ,np.linspace(-1,1,21),cmap=cmap)
    ax1.set_title('mode1 (%s'%(round(pers[0],2))+"%)",loc ='left')
    
    ax2 = fig.add_subplot(3,2, 2)
    ax2.plot(year,PCs[:,0] ,color='k',linewidth=1.2,linestyle='--')
    #print(np.polyfit(range(len(PCs[:,0])),PCs[:,0],1))
    y1=np.poly1d(np.polyfit(year,PCs[:,0],1))
    ax2.plot(year,y1(year),'k--',linewidth=1.2)
    b=ax2.bar(year,PCs[:,0] ,color='r')
    #对y值大于0设置为蓝色  小于0的柱设置为绿色
    for bar,height in zip(b,PCs[:,0]):
        if height<0:
            bar.set(color='blue')
    ax2.set_title('PC1'%(round(pers[0],2)),loc ='left')
    
    ax3 = fig.add_subplot(3,2, 3, projection=proj)
    mapart(ax3, root_path)
    EOF1[A[0]] = EOFs[1]
    pp = ax3.contourf(lons,lats,EOF1 ,np.linspace(-1,1,21),cmap=cmap)
    ax3.set_title('mode2 (%s'%(round(pers[1],2))+"%)",loc ='left')
     
    ax4 = fig.add_subplot(3,2, 4)
    ax4.plot(year,PCs[:,1] ,color='k',linewidth=1.2,linestyle='--')
    ax4.set_title('PC2'%(round(pers[1],2)),loc ='left')
    # print(np.polyfit(year,PCs[:,1],1))
    y2=np.poly1d(np.polyfit(year,PCs[:,1],1))
    #print(y2)
    ax4.plot(year,y2(year),'k--',linewidth=1.2)
    
    bb=ax4.bar(year,PCs[:,1] ,color='r')
    #对y值大于0设置为蓝色  小于0的柱设置为绿色
    for bar,height in zip(bb,PCs[:,1]):
        if height<0:
            bar.set(color='blue')
    
    ax5 = fig.add_subplot(3,2, 5, projection=proj)
    mapart(ax5, root_path)
    EOF1[A[0]] = EOFs[2]
    ppp = ax5.contourf(lons,lats,EOF1 ,np.linspace(-1,1,21),cmap=cmap)
    ax5.set_title('mode3 (%s'%(round(pers[2],2))+"%)",loc ='left')
    
    ax6 = fig.add_subplot(3,2, 6)
    ax6.plot(year,PCs[:,2] ,color='k',linewidth=1.2,linestyle='--')
    ax6.set_title('PC3'%(round(pers[2],2)),loc ='left')
    
    y3=np.poly1d(np.polyfit(year,PCs[:,2],1))
    #print(y3)
    ax6.plot(year,y3(year),'k--',linewidth=1.2)
    
    bbb=ax6.bar(year,PCs[:,2] ,color='r')
    #对y值大于0设置为蓝色  小于0的柱设置为绿色
    for bar,height in zip(bbb,PCs[:,2]):
        if height<0:
            bar.set(color='blue')
    
    #添加0线
    ax2.axhline(y=0,  linewidth=1, color = 'k',linestyle='-')
    ax4.axhline(y=0,  linewidth=1, color = 'k',linestyle='-')
    ax6.axhline(y=0,  linewidth=1, color = 'k',linestyle='-')
            
    #在图下边留白边放colorbar        
    fig.subplots_adjust(bottom=0.1)
    #colorbar位置： 左 下 宽 高 
    l = 0.25
    b = 0.04
    w = 0.6
    h = 0.015
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect) 
    
    c=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal', aspect=20, pad=0.2)
    c.ax.tick_params(labelsize=14)
    #c.set_label('%s'%(labelname),fontsize=20)
    #c.set_ticks(np.arange(1,6,1))
    plt.suptitle(title)   
    plt.subplots_adjust( wspace=0.1,hspace=0.2)
    # plt.tight_layout()
    # plt.savefig('eof_%s.jpg'%name,dpi=300,format='jpg',bbox_inches = 'tight',transparent=True, pad_inches = 0)
    plt.show()


def draw_wavelet(spei, period, power, sig95, coi, global_ws, global_signif, time, xlim,):
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(9, 6),dpi=250)
    gs = GridSpec(2, 4, hspace=0.4, wspace=0.75)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                        wspace=0, hspace=0)
    ax = plt.subplot(gs[0, 0:3])
    # plt.bar(time, spei, 'k')
    
    b=plt.bar(time,spei, color='b', width=0.4, alpha=0.5)
    #对y值大于0设置为蓝色  小于0的柱设置为绿色
    for bar,height in zip(b,spei):
        if height<0:
            bar.set(color='orangered', alpha=0.5)
            
    plt.xlim(xlim[:])
    plt.ylim([-2,2])
    plt.xlabel('Time (year)')
    plt.ylabel('SPEI')
    plt.title('a) SPEI for area average')
    from matplotlib.ticker import MultipleLocator
    xmajorLocator   = MultipleLocator(5) #将x主刻度标签设置为a的倍数
    xminorLocator   = MultipleLocator(1) #将x轴次刻度标签设置为n的倍数
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)


    # --- Contour plot wavelet power spectrum
    # plt3 = plt.subplot(3, 1, 2)
    plt3 = plt.subplot(gs[1, 0:3])
    levels = [0, 0.5, 1, 2, 4, 999]
    # *** or use 'contour'
    CS = plt.contourf(time, period, power, len(levels))
    im = plt.contourf(CS, levels=levels,colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
    plt.xlabel('Time (year)')
    plt.ylabel('Period (years)')
    plt.title('b) Wavelet Power Spectrum (contours at 0.5,1,2,4)')
    plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
        edgecolor="#00000040", hatch='x')
    plt.plot(time, coi, 'k')
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    
    # --- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period)
    plt.plot(global_signif, period, '--')
    plt.xlabel('Power')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()
    plt.savefig('myvave.png', dpi=300)
    plt.show()
    
    
# if __name__ == "__main__":
#     GIMMS = r"E:\climate data\Climate station data\chazhi_out\prec_temp_1960_1961_daily.nc"
#     ndvi_a, time_a, lon, lat = read_nc(GIMMS, "Tmean", scale=1)
#     ndvi_a[ndvi_a == -32768] = np.NAN
#     plot_area_sig(ndvi_a[2], lon, lat, vmin=-30, vmax=30, title='1')

