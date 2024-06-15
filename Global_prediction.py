#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
#python中赋值，如果是可变对象，对其中一个修改会影响到另一个。如果要生成完全新的对象，应使用deepcopy
import joblib
import time


# In[2]:


scaler_CH4 = joblib.load('scaler_CH4.pkl')
scaler_N2O = joblib.load('scaler_N2O.pkl')
scaler_yield = joblib.load('scaler_yield.pkl')
xgbr_CH4=joblib.load("xgbr_CH4.pkl")
xgbr_N2O=joblib.load("xgbr_N2O.pkl")
xgbr_yield=joblib.load("xgbr_yield.pkl")


# In[3]:


attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']


# In[4]:


rice=pd.read_csv(r"D:\li\codesdata\rice\data\rice_attr_new.csv")


# In[5]:


rice.describe()


# In[6]:


def data_prepare(rice):
    rice['duration2'].fillna(0, inplace=True)
    rice.fillna(rice.interpolate(),inplace=True)
    rice.rename(columns={'C_N':'C/N'},inplace=True)
    rice['rainfed_fraction']=rice['rainfed']/rice['area']
    rice['irrigated_fraction']=rice['irrigated']/rice['area']
    return rice
rice=pd.read_csv(r"D:\li\codesdata\rice\data\rice_attr.csv")
rice=data_prepare(rice)
rice=round(rice,2)
rice.rename(columns={rice.columns[0]:'longitude'},inplace=True)
rice.rename(columns={rice.columns[1]:'latitude'},inplace=True)


# In[7]:


rice.loc[rice['duration2'] != 0].describe()


# In[16]:


# attributes = ['temp','prec', 'duration', 'density', 'clay', 'totN', 'TOC',
#        'C/N', 'pH', 'irrigated', 'rainfed', 'area', 'Norg', 'Nin', 'Nmanure',
#        'Nresidue', 'AWD', 'CF', 'MSD', 'physical_area', 'rainfed_fraction',
#        'irrigated_fraction']

# attributes2=['temp','prec',  'duration2', 'density', 'clay', 'totN', 'TOC',
#        'C/N', 'pH', 'irrigated', 'rainfed', 'area', 'Norg', 'Nin', 'Nmanure',
#        'Nresidue', 'AWD', 'CF', 'MSD', 'physical_area', 'rainfed_fraction',
#        'irrigated_fraction']


# In[17]:


# s=copy.deepcopy(rice)
# s = rice[attributes]
# column_names = ['temp','prec', 'duration', 'density', 'clay', 'totN', 'TOC',
#        'C/N', 'pH', 'irrigated', 'rainfed', 'area', 'Norg', 'Nin', 'Nmanure',
#        'Nresidue', 'AWD', 'CF', 'MSD', 'physical_area', 'rainfed_fraction',
#        'irrigated_fraction']
# s.columns = column_names


# In[8]:


attributes=['duration', 'temp', 'prec',
           'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
           'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
           'water_regime_rainfed']
attributes2 = ['duration2', 'temp', 'prec',
           'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
           'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
           'water_regime_rainfed']

def rice_cal(rice,model,scaler,attributes_):
    
    start =time.time()
    
    rice['water_regime_AWD']=rice['AWD']*rice['irrigated_fraction']
    rice['water_regime_continuously flooding']=rice['CF']*rice['irrigated_fraction']
    rice['water_regime_midseason drainage']=rice['MSD']*rice['irrigated_fraction']
    rice['water_regime_rainfed']=rice['rainfed_fraction']
    
    X = rice[attributes_]
    X_copy = X.copy()

    # 将第一列的列名更改为'duration'
    X_copy.rename(columns={X_copy.columns[0]: 'duration'}, inplace=True)
    # 使用副本进行操作，避免警告
    X_copy.iloc[:, :11] = scaler_CH4.transform(X_copy.iloc[:, :11])
    
    
    pred = model.predict(X_copy)
#     s=copy.deepcopy(rice)
#     s = rice[attributes]
#     column_names = ['temp','prec', 'duration', 'density', 'clay', 'totN', 'TOC',
#        'C/N', 'pH', 'irrigated', 'rainfed', 'area', 'Norg', 'Nin', 'Nmanure',
#        'Nresidue', 'AWD', 'CF', 'MSD', 'physical_area', 'rainfed_fraction',
#        'irrigated_fraction']
#     s.columns = column_names
#     #pred different attr values
#     #AWD

#     s.loc[:, 'water_regime_AWD'] = 1
#     s.loc[:, 'water_regime_continuously flooding']=0
#     s.loc[:, 'water_regime_midseason drainage']=0
#     s.loc[:, 'water_regime_rainfed']=0
#     rice_AWD=copy.deepcopy(s)

#     #CF
#     s.loc[:, 'water_regime_AWD'] =0 
#     s.loc[:, 'water_regime_continuously flooding']=1
#     s.loc[:, 'water_regime_midseason drainage']=0
#     s.loc[:, 'water_regime_rainfed']=0
#     rice_CF=copy.deepcopy(s)
    
#     #MSD
#     s.loc[:, 'water_regime_AWD'] = 0
#     s.loc[:, 'water_regime_continuously flooding']=0
#     s.loc[:, 'water_regime_midseason drainage']= 1
#     s.loc[:, 'water_regime_rainfed'] = 0
#     rice_MSD=copy.deepcopy(s)
    
#     # rainfed
#     s.loc[:, 'water_regime_AWD'] = 0
#     s.loc[:, 'water_regime_continuously flooding']=0
#     s.loc[:, 'water_regime_midseason drainage']=0
#     s.loc[:, 'water_regime_rainfed'] = 1
#     rice_rainfed=copy.deepcopy(s)
    
#     num_attr = ['duration', 'temp', 'prec',
#        'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin']
#     for k in (rice_AWD,rice_CF,rice_MSD,rice_rainfed):
#         # 创建列名映射字典
#         k[num_attr]=scaler.transform(k[num_attr])
        
    
    
#     attributes = ['duration', 'temp', 'prec',
#        'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
#        'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
#        'water_regime_rainfed']
#     pred_AWD = model.predict(rice_AWD[attributes])
#     pred_CF = model.predict(rice_CF[attributes])
#     pred_MSD = model.predict(rice_MSD[attributes])
#     pred_rainfed = model.predict(rice_rainfed[attributes])

    
#     pred = pred_AWD*rice['AWD']*rice['irrigated_fraction']+pred_CF*rice['CF']*rice['irrigated_fraction']+ pred_MSD*rice['MSD']*rice['irrigated_fraction']+pred_rainfed*rice['rainfed_fraction']

    return pred


# In[9]:


def rice_cal_tot(rice):
    start =time.time()
    rice['CH4_xgb_per_area']=rice_cal(rice,xgbr_CH4,scaler_CH4,attributes)
    rice['CH4_xgb_per_area2']=rice_cal(rice,xgbr_CH4,scaler_CH4,attributes2)
    rice.loc[rice['duration2'] == 0, 'CH4_xgb'] = rice['CH4_xgb_per_area']*rice['area']
    rice.loc[rice['duration2'] != 0, 'CH4_xgb'] = rice['CH4_xgb_per_area']*rice['physical_area']+ rice['CH4_xgb_per_area2']*(rice['area']-rice['physical_area'])
#     rice['CH4_xgb'] = rice['CH4_xgb_per_area']*rice['area']
    
    rice['N2O_xgb_per_area']=rice_cal(rice,xgbr_N2O,scaler_N2O,attributes)
    rice['N2O_xgb_per_area2']=rice_cal(rice,xgbr_N2O,scaler_N2O,attributes2)
    rice.loc[rice['duration2'] == 0, 'N2O_xgb'] = rice['N2O_xgb_per_area']*rice['area']
    rice.loc[rice['duration2'] != 0, 'N2O_xgb'] = rice['N2O_xgb_per_area']*rice['physical_area']+rice['N2O_xgb_per_area2']*(rice['area']-rice['physical_area'])


    
    rice['yield_xgb_per_area']=rice_cal(rice,xgbr_yield,scaler_yield,attributes)
    rice['yield_xgb_per_area2']=rice_cal(rice,xgbr_yield,scaler_yield,attributes2)
    rice.loc[rice['duration2'] == 0, 'yield_xgb'] = rice['yield_xgb_per_area']*rice['area']
    rice.loc[rice['duration2'] != 0, 'yield_xgb'] = rice['yield_xgb_per_area']*rice['physical_area']+rice['yield_xgb_per_area2']*(rice['area']-rice['physical_area'])



    end = time.time()
    print('Running time: %s Seconds'%(end-start))
    return rice


# In[10]:


rice=rice_cal_tot(rice)


# In[69]:


sum(rice['CH4_xgb'])#10^9kg


# In[14]:


sum(rice['CH4_xgb'])#10^9kg


# In[15]:


sum(rice['N2O_xgb'])


# In[16]:


sum(rice['yield_xgb'])


# In[17]:


joblib.dump(rice,"rice.pkl")


# In[18]:


rice.describe()


# # Fig2 global prediction

# In[4]:


# 导入经纬度
lonlat = pd.read_csv(r"D:\li\codesdata\rice\data\lonlat.csv")
lonlat.rename(columns={lonlat.columns[0]:'longitude'},inplace=True)
lonlat.rename(columns={lonlat.columns[1]:'latitude'},inplace=True)
lonlat=round(lonlat,2)


# In[28]:


lonlat


# In[29]:


start=time.time()
a=pd.merge(lonlat,rice,how='left',on=['longitude','latitude'])
CH4 = np.array(a['CH4_xgb_per_area']).reshape(2160,4320)
N2O = np.array(a['N2O_xgb_per_area']).reshape(2160,4320)
yields = np.array(a['yield_xgb_per_area']).reshape(2160,4320)

CH4_total = np.array(a['CH4_xgb']).reshape(2160,4320)
N2O_total = np.array(a['N2O_xgb']).reshape(2160,4320)
yields_total = np.array(a['yield_xgb']).reshape(2160,4320)

preds = [CH4,N2O,yields]
pred_totals = [CH4_total,N2O_total,yields_total]

end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[31]:


np.save("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/global_prediction.npy",preds)


# In[45]:


np.save("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/baseline_global_prediction_total.npy",pred_totals)


# In[13]:


preds = np.load("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/global_prediction.npy")
pred_totals = np.load("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/baseline_global_prediction_total.npy")


# In[49]:


def arr2raster(arr, raster_file, prj=None, trans=None):
    """
    将数组转成栅格文件写入硬盘
    :param arr: 输入的mask数组 ReadAsArray()
    :param raster_file: 输出的栅格文件路径
    :param prj: gdal读取的投影信息 GetProjection()，默认为空
    :param trans: gdal读取的几何信息 GetGeoTransform()，默认为空
    :return:
    """
 
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, np.array(arr).shape[2], np.array(arr).shape[1], 3, gdal.GDT_Float32)
 
    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)
 
    # 将数组的各通道写入图片
    dst_ds.GetRasterBand(1).WriteArray(arr[0])
    dst_ds.GetRasterBand(2).WriteArray(arr[1])
    dst_ds.GetRasterBand(3).WriteArray(arr[2])
    
    dst_ds.FlushCache()
    dst_ds = None
    print("successfully convert array to raster")


src_ras_file = r"D:\li\codesdata\rice\data\SPAM\spam2000v3.0.7_global_physical-area.geotiff\spam2000v3r7_physical-area_RICE.tif"  
# 提供地理坐标信息和几何信息的栅格底图
dataset = gdal.Open(src_ras_file)
projection = dataset.GetProjection()
transform = dataset.GetGeoTransform()
raster_file = r"D:\li\codesdata\rice\python_calculate\NEW_V2\future_prediction_npy\global_prediction_total.tif"
arr2raster(pred_totals, raster_file, prj=projection, trans=transform)


# In[6]:


import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
# from skimage import io
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
#import fiona
import shapefile
import time
import seaborn as sns
from adjustText import adjust_text
import matplotlib
from matplotlib import colors
import joblib
# import fiona
import pyogrio# pyogrio-0.5.1
# import cartopy


# In[7]:


# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 12, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['Arial'], # 'Simsun'宋体
    "font.sans-serif":['Microsoft YaHei'],
    "axes.unicode_minus": False,# 用来正常显示负号
}
plt.rcParams.update(config)


# In[8]:


def getdata(filepath,band):
    dataset=rasterio.open(filepath)
    data = dataset.read(band)
    data[data<-9999]=np.nan
    return data
areapath="D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"
area=getdata(areapath,1)
area=np.where(area<0,0,area)


# In[9]:


def bootstrap_ci(data,alpha):
    lows=[]
    ups=[]
    
    for i in range(data.shape[0]):
        subdata=data[i,:]
        subdata=pd.DataFrame(subdata).dropna()
        
        #计算原始样本均值
        ori_mean = np.nanmean(subdata)
        data_boot = []
        num=len(subdata)
        for b in range(num):
            boot_mean = resample(subdata, replace=True, n_samples=num).mean()
            data_boot.append(boot_mean[0])
        sample = pd.DataFrame({'经验均值':data_boot-ori_mean})
        low = min(ori_mean - sample.经验均值.quantile((1-alpha)/2),
                      ori_mean - sample.经验均值.quantile(alpha+(1-alpha)/2))
        up = max(ori_mean - sample.经验均值.quantile((1-alpha)/2),
                     ori_mean - sample.经验均值.quantile(alpha+(1-alpha)/2))
        lows.append(low)
        ups.append(up)        
    result=pd.DataFrame([lows,ups])
    return(result)


# In[10]:


import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")


# In[18]:


# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 18, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['Arial'], # 'Simsun'宋体
    "font.sans-serif":['Microsoft YaHei'],
    "axes.unicode_minus": False,# 用来正常显示负号
}
plt.rcParams.update(config)


# In[19]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.utils import resample
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

fig, axs = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'width_ratios': [3, 1, 1]})

# Define bounds and colormap
bounds_list = [
    [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    [0, 0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
]
labels = [
    "C$\mathregular{H_4}$ (kg ha$\mathregular{^{-1}}$)", 
    "$\mathregular{N_2}$O (kgN ha$\mathregular{^{-1}}$)", 
    'Yield (t ha$^{-1}$)'
]
cmap = plt.get_cmap('Spectral_r', 10)

# Plot each subplot
for i, (pred, bounds) in enumerate(zip(preds, bounds_list)):
    x = np.linspace(-180, 180, pred.shape[1])
    y = np.linspace(90, -90, pred.shape[0])
    xx, yy = np.meshgrid(x, y)

    norm = colors.BoundaryNorm(bounds, 10)

    # Plot the map
    m = Basemap(ax=axs[i,0], llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawlsmask(land_color='1')
    m.pcolor(xx, yy, pred, latlon=True, cmap=cmap, norm=norm)

    # Add colorbar below the map
    divider = make_axes_locatable(axs[i,0])
    cax = divider.append_axes("bottom", size="5%", pad=0.2)  # Adjust the pad value as needed
    cb = matplotlib.colorbar.ColorbarBase(cax, norm=norm, ticks=bounds, cmap=cmap, orientation='horizontal', extend='neither')
    cax.set_xticks(bounds)
    cax.set_xticklabels(bounds)

    # Add title
    title_mapping = {0:"C$\mathregular{H_4}$ (kg ha$\mathregular{^{-1}}$)", 
                     1: "$\mathregular{N_2}$O (kgN ha$\mathregular{^{-1}}$)", 
                     2: 'Yield (t ha$^{-1}$)'}
    cb.set_label(title_mapping[i], fontsize=18)
    # 去除边框
    axs[i, 0].spines['top'].set_visible(False)
    axs[i, 0].spines['bottom'].set_visible(False)
    axs[i, 0].spines['left'].set_visible(False)
    axs[i, 0].spines['right'].set_visible(False)

# 第二列平均值
for i, (pred, label) in enumerate(zip(preds, labels)):
    ax = axs[i, 1]
    ax.set_yticks([30,20,10,0,-10,-20,-30,-40,-50])

    ax.set_yticklabels(['30$^\circ$S','20$^\circ$S','10$^\circ$S','0$^\circ$','10$^\circ$N','20$^\circ$N','30$^\circ$N','40$^\circ$N','50$^\circ$N'])
    ax.invert_yaxis()#翻转y轴
    ax.set_xlabel(label)
    ax.yaxis.tick_right()
    x=np.linspace(-90,90,num=2160)
    y=np.nanmean(pred,axis=1)
#     yest=lowess(y,x,frac=0.1)
    
#     ax.plot(yest[:,1],yest[:,0],linewidth=5,color='red', linestyle='-')
    ax.plot(y,x,color='r', linestyle='-')
    start=time.time()
    ci_bound=bootstrap_ci(pred,0.95)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))
    ax.fill_betweenx(x,ci_bound.iloc[0],ci_bound.iloc[1],alpha=0.5,facecolor ='grey')
    ax.set_ylim(40,-60)
#     ax.set_aspect(aspect=2)
# 第三列面积加权平均
labels = [
    "C$\mathregular{H_4}$ (Tg)", 
    "$\mathregular{N_2}$O (tN)", 
    'Yield (Mt)'
]
for i, (pred_total, label) in enumerate(zip(pred_totals, labels)):
    ax = axs[i, 2]
    ax.set_yticks([30,20,10,0,-10,-20,-30,-40,-50])
    ax.set_yticklabels(['30$^\circ$S','20$^\circ$S','10$^\circ$S','0$^\circ$','10$^\circ$N','20$^\circ$N','30$^\circ$N','40$^\circ$N','50$^\circ$N'])
    ax.invert_yaxis()#翻转y轴
    ax.set_xlabel(label)
    ax.yaxis.tick_right()
    x=np.linspace(-90,90,num=2160)
    sumdata = pred_total
    y=np.nansum(sumdata,axis=1)
    if i==0: 
        y = y/1e9
    elif i==1:
        y=y/1e3
    else:
        y=y/1e6
#     yest=lowess(y,x,frac=0.1)
#     ax.plot(yest[:,1],yest[:,0],linewidth=5,color='red', linestyle='-')
    ax.plot(y,x,color='r', linestyle='-')
    ax.set_ylim(40,-60)
#     ax.set_aspect(aspect=2)
plt.tight_layout()
axs[0,0].set_title("a",loc='left',fontweight='bold',fontsize=25)
axs[1,0].set_title("b",loc='left',fontweight='bold',fontsize=25)
axs[2,0].set_title("c",loc='left',fontweight='bold',fontsize=25)
axs[0,1].set_title("d",loc='left',fontweight='bold',fontsize=25)
axs[1,1].set_title("e",loc='left',fontweight='bold',fontsize=25)
axs[2,1].set_title("f",loc='left',fontweight='bold',fontsize=25)
axs[0,2].set_title("g",loc='left',fontweight='bold',fontsize=25)
axs[1,2].set_title("h",loc='left',fontweight='bold',fontsize=25)
axs[2,2].set_title("i",loc='left',fontweight='bold',fontsize=25)
# axs[1].set_title("b",loc='left',fontweight='bold',fontsize=15)
# axs[2].set_title("c",loc='left',fontweight='bold',fontsize=15)
# # Show the plot
plt.show()


# In[20]:


fig.savefig(r'E:\li\文章\审稿人意见\李润桐-NF-0810\figs\global_pred_0609.png', dpi=600, bbox_inches='tight')


# # Fig3 SHAP

# In[9]:


import shap


# In[151]:


rice = joblib.load('rice.pkl')


# In[152]:


rice['water_regime_AWD']=rice['AWD']*rice['irrigated_fraction']
rice['water_regime_continuously flooding']=rice['CF']*rice['irrigated_fraction']
rice['water_regime_midseason drainage']=rice['MSD']*rice['irrigated_fraction']
rice['water_regime_rainfed']=rice['rainfed_fraction']
attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']


# In[153]:


num_attr = ['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin']


# In[154]:


# 
model_CH4=xgbr_CH4
start =time.time()
X = rice[attributes]
X[num_attr] = scaler_CH4.transform(X[num_attr])
explainer_CH4 = shap.Explainer(model_CH4)
columns=['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'AWD','CF','MSD','Rainfed']
X.columns = columns
shap_values_CH4 = explainer_CH4(X)

# start=time.time()

shap.summary_plot(shap_values_CH4, X,show = False)
plt.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\summary_plot_CH4.png",dpi=600,bbox_inches='tight')
# end = time.time()
# print('Running time: %s Seconds'%(end-start))

end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[155]:


model_N2O=xgbr_N2O
start =time.time()
X = rice[attributes]
X[num_attr] = scaler_N2O.transform(X[num_attr])
explainer_N2O = shap.Explainer(model_N2O)
columns=['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'AWD','CF','MSD','Rainfed']
X.columns = columns
shap_values_N2O = explainer_N2O(X)

shap.summary_plot(shap_values_N2O, X,show = False)
plt.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\summary_plot_N2O.png",dpi=600,bbox_inches='tight')

end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[156]:


model_yield=xgbr_yield
start =time.time()
X = rice[attributes]
X[num_attr] = scaler_yield.transform(X[num_attr])
explainer_yield = shap.Explainer(model_yield)
columns=['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'AWD','CF','MSD','Rainfed']
X.columns = columns
shap_values_yield = explainer_yield(X)

shap.summary_plot(shap_values_yield, X,show = False)
plt.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\summary_plot_yield.png",dpi=600,bbox_inches='tight')

end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[157]:


def shap_out(shap_values):
    x=pd.DataFrame(shap_values.values)
    x=abs(x)
    c=pd.DataFrame(x.idxmax(1),columns=['shap'])
    c['x']=rice['longitude']
    c['y']=rice['latitude']
    return c


# In[158]:


shap_CH4_out=shap_out(shap_values_CH4)
shap_N2O_out=shap_out(shap_values_N2O)
shap_yield_out=shap_out(shap_values_yield)


# In[159]:


shap_N2O_out.describe()


# In[160]:


# 使用 rename 方法更改列名
# 导入经纬度
lonlat = pd.read_csv(r"D:\li\codesdata\rice\data\lonlat.csv")
# lonlat.rename(columns={lonlat.columns[0]:'longitude'},inplace=True)
# lonlat.rename(columns={lonlat.columns[1]:'latitude'},inplace=True)
lonlat=round(lonlat,2)
# lonlat.rename(columns={'longitude': 'x', 'latitude': 'y'}, inplace=True)


# In[161]:


start=time.time()
a=pd.merge(lonlat,shap_CH4_out,how='left',on=['x','y'])
shap_CH4_xgb=np.array(a['shap']).reshape(2160,4320)

a=pd.merge(lonlat,shap_N2O_out,how='left',on=['x','y'])
shap_N2O_xgb=np.array(a['shap']).reshape(2160,4320)

a=pd.merge(lonlat,shap_yield_out,how='left',on=['x','y'])
shap_yield_xgb=np.array(a['shap']).reshape(2160,4320)

shap_xgb=[shap_CH4_xgb,shap_N2O_xgb,shap_yield_xgb]
end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[10]:


from osgeo import gdal
from osgeo import gdal, osr
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import joblib
from matplotlib import colors
import matplotlib.gridspec as gridspec
import os
# from osgeo import gdal
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors


# In[163]:


def arr2raster(arr, raster_file, prj=None, trans=None):
    """
    将数组转成栅格文件写入硬盘
    :param arr: 输入的mask数组 ReadAsArray()
    :param raster_file: 输出的栅格文件路径
    :param prj: gdal读取的投影信息 GetProjection()，默认为空
    :param trans: gdal读取的几何信息 GetGeoTransform()，默认为空
    :return:
    """
 
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, np.array(arr).shape[2], np.array(arr).shape[1], 3, gdal.GDT_Float32)
 
    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)
 
    # 将数组的各通道写入图片
    dst_ds.GetRasterBand(1).WriteArray(arr[0])
    dst_ds.GetRasterBand(2).WriteArray(arr[1])
    dst_ds.GetRasterBand(3).WriteArray(arr[2])
    
    dst_ds.FlushCache()
    dst_ds = None
    print("successfully convert array to raster")


# In[164]:


src_ras_file = r"D:\li\codesdata\rice\data\SPAM\spam2000v3.0.7_global_physical-area.geotiff\spam2000v3r7_physical-area_RICE.tif"  
# 提供地理坐标信息和几何信息的栅格底图
dataset = gdal.Open(src_ras_file)
projection = dataset.GetProjection()
transform = dataset.GetGeoTransform()
raster_file = r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\shap_xgb.tif"
arr2raster(shap_xgb, raster_file, prj=projection, trans=transform)


# In[166]:


shap_CH4_xgb = getdata(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\shap_xgb.tif",1)
shap_N2O_xgb = getdata(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\shap_xgb.tif",2)
shap_yield_xgb = getdata(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\shap_xgb.tif",3)
areapath = r"D:\li\codesdata\rice\data\SPAM\spam2000v3.0.7_global_harvested-area.geotiff\spam2000v3r7_harvested-area_RICE.tif"
area = getdata(areapath,1)


# In[ ]:


np.nansum(area)


# In[117]:


# 计算有机氮肥为最重要驱动因素的比例
areas = []
for k in range(11):
    data_k = np.where(shap_CH4_xgb==k,1,np.nan)
    feature_area=data_k*area
    areas.append(np.nansum(feature_area))

water_regime = np.where(shap_CH4_xgb>10,1,np.nan)
water_regime_area = water_regime*area
areas.append(np.nansum(water_regime_area))


# In[186]:


attributes


# In[119]:


areas/np.sum(areas)


# In[121]:


areas = []
for k in range(11):
    data_k = np.where(shap_N2O_xgb==k,1,np.nan)
    feature_area=data_k*area
    areas.append(np.nansum(feature_area))

water_regime = np.where(shap_N2O_xgb>10,1,np.nan)
water_regime_area = water_regime*area
areas.append(np.nansum(water_regime_area))


# In[122]:


areas/np.nansum(areas)


# In[127]:


areas = []
for k in range(11):
    data_k = np.where(shap_yield_xgb==k,1,np.nan)
    feature_area=data_k*area
    areas.append(np.nansum(feature_area))

water_regime = np.where(shap_yield_xgb>10,1,np.nan)
water_regime_area = water_regime*area
areas.append(np.nansum(water_regime_area))
areas/np.nansum(areas)


# In[128]:


attributes


# In[39]:


features = pd.read_csv(r'E:\li\文章\审稿人意见\features.csv')


# In[48]:


features


# In[53]:


features['N2O'][0]


# In[167]:


##特征重要性，每个特征为主要因素所占的面积比例
def feature_importance_percentage_per_country(i):
    shpPath = filenames[i]
    imgPath = r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\shap_xgb.tif"
    areapath = r"D:\li\codesdata\rice\data\SPAM\spam2000v3.0.7_global_harvested-area.geotiff\spam2000v3r7_harvested-area_RICE.tif"
    data=clip(shpPath, imgPath)
    area=clip(shpPath, areapath) # 收获面积
    tot_area = np.nansum(area)
    areas1=[]
    areas2=[]
    areas3=[]
    for k in range(11):
        data_k = np.where(data==k,1,np.nan)
        feature_area=data_k*area
        areas1.append(np.nansum(feature_area[0]))
        areas2.append(np.nansum(feature_area[1]))
        areas3.append(np.nansum(feature_area[2]))
    water_regime = np.where(data>10,1,np.nan)
    water_regime_area = water_regime*area
    areas1.append(np.nansum(water_regime_area[0]))
    areas2.append(np.nansum(water_regime_area[1]))
    areas3.append(np.nansum(water_regime_area[2]))
    areas1 = np.round([x/np.sum(areas1) for x in areas1],3)
    areas2 = np.round([x/np.sum(areas2) for x in areas2],3)
    areas3 = np.round([x/np.sum(areas3) for x in areas3],3)
    
    shps = shapefile.Reader(shpPath)
    abb = shps.shapeRecord().record[0]
    country = shps.shapeRecord().record[1]
    result = [abb,country,tot_area,areas1,areas2,areas3]
    
    return(result)


# In[168]:


def clip(shpPath, imagePath):
    """
    根据传入的矢量数据裁剪栅格影像
    :param shpPath: 矢量数据
    :param imagePath: 栅格影像
    :param outImagePath: 裁剪后的影像
    :return:
    """
    shpData = gpd.read_file(shpPath)
    # 列表推导式
    # __geo_interface__ ：转成地理接口
    shapes = [shpData.geometry[i].__geo_interface__ for i in range(len(shpData))]

    # 读取输入图像
    with rasterio.open(imagePath) as src:
        out_image, out_transform = mask(src, shapes, crop=True, nodata=np.nan)
        out_meta = src.meta

    # 更新元数据
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    return(out_image)

#     # 输出掩膜提取图像
#     with rasterio.open(outImagePath, "w", **out_meta) as dest:
#         dest.write(out_image)


# In[169]:


files=os.listdir('D:/li/mapdata/countrydata')
filenames=[]
for i in range(len(files)):
    path=os.path.join('D:/li/mapdata/countrydata/'+files[i]+'/')
    #paths.append(path)
    name=os.path.join(files[i][0:-4]+'_0.shp')
    filename=os.path.join(path+name)
    filenames.append(filename)


# In[170]:


importance=[]
for i in range(len(filenames)):
    start =time.time()
    s=feature_importance_percentage_per_country(i)
    importance.append(s)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))
features=pd.DataFrame(importance)
features.columns=['abb','country','area','CH4','N2O','yields']


# In[171]:


sorted_features = features.sort_values(by='area', ascending=False)
sorted_features.reset_index(drop=True, inplace=True)


# In[184]:


sorted_features['N2O'][0]


# In[185]:


sorted_features['N2O'][1]


# In[173]:


features = sorted_features
features.to_csv(r'E:\li\文章\审稿人意见\features.csv')


# In[174]:


cols=['#bf812d','#fee08b','#a6d96a','#1f78b4','#fb9a99','#4eb3d3','#33a02c','#ff7f00','#cab2d6','#CD5C5C','#6a3d9a','#ffff99']
cmap = colors.ListedColormap(cols)


# In[16]:


config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 12, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['Arial'], # 'Simsun'宋体
    "axes.unicode_minus": False,# 用来正常显示负号
}
plt.rcParams.update(config)


# In[176]:


filepath = r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\shap_xgb.tif"
fig=plt.figure(figsize=(10,6))
##rect可以设置子图的位置与大小
rect1 = [0, 0.95, 0.33, 0.5] 
# [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例），左下角为(0,0),右上角为(1,1)
rect2 = [0.35, 0.95, 0.33, 0.5]
rect3 = [0.7, 0.95, 0.33, 0.5]
rect4 = [0, 0.7, 0.33, 0.3] 
rect5 = [0.35, 0.7, 0.33, 0.3]
rect6 = [0.7, 0.7, 0.33, 0.3]
index=[i for i in range(len(features))]
data1=getdata(filepath,1)#XGB CH4
data2=getdata(filepath,2)#XGB N2O
data3=getdata(filepath,3)#XGB yield

x = np.linspace(-180,180, data1.shape[1])
y = np.linspace(90,-90, data1.shape[0])

labels=['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'Water regime']
xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵
cols=['#bf812d','#fee08b','#a6d96a','#1f78b4','#fb9a99','#4eb3d3',
      '#33a02c','#ff7f00','#cab2d6','#CD5C5C','#6a3d9a','#ffff99']
cmap = colors.ListedColormap(cols)
bounds=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,14]
norm = colors.BoundaryNorm(bounds, cmap.N)



#在fig中添加子图ax，并赋值位置rect
ax1 = plt.axes(rect1)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90,ax=ax1)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawcoastlines(linewidth=0.5)
m.drawlsmask(land_color='1')
m.pcolor(xx, yy, data1,latlon=True,cmap=cmap,norm=norm)


ax2 = plt.axes(rect2)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90,ax=ax2)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawcoastlines(linewidth=0.5)
m.drawlsmask(land_color='1')

m.pcolor(xx, yy, data2,latlon=True,cmap=cmap,norm=norm)


ax3 = plt.axes(rect3)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90,ax=ax3)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawcoastlines(linewidth=0.5)
m.drawlsmask(land_color='1')
m.pcolor(xx, yy, data3,latlon=True,cmap=cmap,norm=norm)


ax4 = plt.axes(rect4)
y=[features['CH4'][i][0] for i in range(len(features))]
y1=[features['CH4'][i][1] for i in range(len(features))]
y = [i*100 for i in y]
y1 = [i*100 for i in y1]

plt.bar(index, y,color=cols[0])
plt.bar(index, y1,color=cols[1],bottom=y)
for j in range(2,len(labels)):
    for k in range(0, len(y)):
        y1[k] = y1[k] + y[k]
    y=[features['CH4'][i][j]*100 for i in range(len(features))]
    
    plt.bar(index, y,color=cols[j],bottom=y1)
        
#plt.xticks(ticks=index)
ax4.set_xticklabels(features['abb'],rotation=45,fontsize=9)
ax4.set_xticks(index)
ax4.set_ylabel('Fraction (%)')
#ax4.set_yticks([])

ax5 = plt.axes(rect5)
y=[features['N2O'][i][0] for i in range(len(features))]
y1=[features['N2O'][i][1] for i in range(len(features))]
y = [i*100 for i in y]
y1 = [i*100 for i in y1]

plt.bar(index, y,color=cols[0])
plt.bar(index, y1,color=cols[1],bottom=y)
plt.ylim(0, 100)
for j in range(2,len(labels)):
    for k in range(0, len(y)):
        y1[k] = y1[k] + y[k]
    y=[features['N2O'][i][j]*100 for i in range(len(features))]
    
    plt.bar(index, y,color=cols[j],bottom=y1)
        
ax5.set_xticklabels(features['abb'],rotation=45,fontsize=9)
ax5.set_xticks(index)
ax5.set_yticks([])

ax6 = plt.axes(rect6)
y=[features['yields'][i][0] for i in range(len(features))]
y1=[features['yields'][i][1] for i in range(len(features))]
y = [i*100 for i in y]
y1 = [i*100 for i in y1]

plt.bar(index, y,color=cols[0])
plt.bar(index, y1,color=cols[1],bottom=y)
for j in range(2,len(labels)):
    for k in range(0, len(y)):
        y1[k] = y1[k] + y[k]
    y=[features['yields'][i][j]*100 for i in range(len(features))]
    
    plt.bar(index, y,color=cols[j],bottom=y1)
        
#plt.xticks(ticks=index)
ax6.set_xticklabels(features['abb'],rotation=45,fontsize=9)
ax6.set_xticks(index)
ax6.set_yticks([])

#添加colorbar
cbar_ax = fig.add_axes([0.05, 0.6, 0.9, 0.02])
width=1/12
ticks=[width*i+width/2 for i in range(0,12)]
matplotlib.colorbar.ColorbarBase(cbar_ax, ticks=ticks,cmap=cmap, orientation='horizontal')
cbar_ax.set_xticklabels(labels)

ax1.set_title("a",loc='left',fontweight='bold',fontsize=15)
ax2.set_title("b",loc='left',fontweight='bold',fontsize=15)
ax3.set_title("c",loc='left',fontweight='bold',fontsize=15)
ax4.set_title("d",loc='left',fontweight='bold',fontsize=15)
ax5.set_title("e",loc='left',fontweight='bold',fontsize=15)
ax6.set_title("f",loc='left',fontweight='bold',fontsize=15)


# In[177]:


y=[features['N2O'][i][10] for i in range(len(features))]


# In[126]:


features


# In[71]:


features['N2O'][9]


# In[129]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\shap_xgb.png",dpi=600,bbox_inches='tight')


# # Supplementary Fig SHAP

# In[56]:


columns=['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'AWD','CF','MSD','Rainfed']
###各属性在空间上SHAP值贡献
spatial_shap_CH4 = pd.DataFrame(shap_values_CH4.values,columns=columns)
spatial_shap_N2O = pd.DataFrame(shap_values_N2O.values,columns=columns)
spatial_shap_yield = pd.DataFrame(shap_values_yield.values,columns=columns)
for c in [spatial_shap_CH4,spatial_shap_N2O,spatial_shap_yield]:
    c['x'] = rice['longitude']
    c['y'] = rice['latitude']


# In[57]:


shap_values_CH4.values.shape


# In[58]:


###各属性在空间上SHAP值贡献
spatial_shap_CH4 = pd.DataFrame(shap_values_CH4.values,columns=columns)
spatial_shap_N2O = pd.DataFrame(shap_values_N2O.values,columns=columns)
spatial_shap_yield = pd.DataFrame(shap_values_yield.values,columns=columns)
for c in [spatial_shap_CH4,spatial_shap_N2O,spatial_shap_yield]:
    c['x'] = rice['longitude']
    c['y'] = rice['latitude']


# In[59]:


for c in [spatial_shap_CH4,spatial_shap_N2O,spatial_shap_yield]:
    c['x'] = rice['longitude']
    c['y'] = rice['latitude']


# In[60]:


###不需要计算特征shap值面积比例，不需要转成tiff，将表格转为数组npy文件
start=time.time()
a=pd.merge(lonlat,spatial_shap_CH4,how='left',on=['x','y'])
shaps_CH4=[]
for i in range(15):
    array = np.array(a.iloc[:,i+3]).reshape(2160,4320)
    shaps_CH4.append(array)   
end = time.time()
print('Running time: %s Seconds'%(end-start))

start=time.time()
a=pd.merge(lonlat,spatial_shap_N2O,how='left',on=['x','y'])
shaps_N2O=[]
for i in range(15):
    array = np.array(a.iloc[:,i+3]).reshape(2160,4320)
    shaps_N2O.append(array)   
end = time.time()
print('Running time: %s Seconds'%(end-start))

start=time.time()
a=pd.merge(lonlat,spatial_shap_yield,how='left',on=['x','y'])
shaps_yield=[]
for i in range(15):
    array = np.array(a.iloc[:,i+3]).reshape(2160,4320)
    shaps_yield.append(array)
end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[61]:


shaps_CH4


# In[62]:


###shap values绘图
fig,axs = plt.subplots(3,4)
fig.set_size_inches(12,6)
x = np.linspace(-180,180, 4320)
y = np.linspace(90,-90, 2160)

labs = ['a','b','c','d','e','f','g','h','i','j','k','l']
titles = ['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'Water regime']

bounds=[-100,-80,-60,-40,-20,0,20,40,60,80,100]


norm = colors.BoundaryNorm(bounds,10)
cmap = plt.get_cmap('PiYG',10)

xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵
for i,ax in enumerate(axs.reshape(-1)):
    m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90,ax=ax)
    if(i==0): 
        m.pcolor(xx,yy,shaps_CH4[0],cmap=cmap,latlon=True,norm=norm) 
    if(i==1): 
        m.pcolor(xx,yy,shaps_CH4[1],cmap=cmap,latlon=True,norm=norm) 
    if(i==2): 
        m.pcolor(xx,yy,shaps_CH4[2],cmap=cmap,latlon=True,norm=norm) 
        
    if(i==3): 
        m.pcolor(xx,yy,shaps_CH4[3],cmap=cmap,latlon=True,norm=norm) 
    if(i==4): 
        m.pcolor(xx,yy,shaps_CH4[4],cmap=cmap,latlon=True,norm=norm) 
    if(i==5): 
        m.pcolor(xx,yy,shaps_CH4[5],cmap=cmap,latlon=True,norm=norm) 

    if(i==6): 
        m.pcolor(xx,yy,shaps_CH4[6],cmap=cmap,latlon=True,norm=norm) 
    if(i==7): 
        m.pcolor(xx,yy,shaps_CH4[7],cmap=cmap,latlon=True,norm=norm) 
    if(i==8): 
        m.pcolor(xx,yy,shaps_CH4[8],cmap=cmap,latlon=True,norm=norm) 
    
    if(i==9): 
        m.pcolor(xx,yy,shaps_CH4[9],cmap=cmap,latlon=True,norm=norm) 
    if(i==10): 
        m.pcolor(xx,yy,shaps_CH4[10],cmap=cmap,latlon=True,norm=norm) 
    if(i==11): 
        data11=shaps_CH4[11]+shaps_CH4[12]+shaps_CH4[13]+shaps_CH4[14]
        m.pcolor(xx,yy,data11,cmap=cmap,latlon=True,norm=norm)   

    m.drawcoastlines()
    m.drawcountries()
    ax.set_title(labs[i],loc='left',fontweight='bold',fontsize=15)
    ax.set_title(titles[i], loc='center')
plt.tight_layout()




cbar_ax = fig.add_axes([0.05, 0, 0.9, 0.02])
cb=matplotlib.colorbar.ColorbarBase(cbar_ax, norm=norm,ticks=bounds,cmap=cmap, orientation='horizontal',extend='neither')
cbar_ax.set_xticks(bounds)
cbar_ax.set_xticklabels(bounds)
cb.set_label("SHAP value")


# In[63]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\spatial_shap_CH4.png",dpi=600,bbox_inches='tight')


# In[64]:


###shap values绘图
fig,axs = plt.subplots(3,4)
fig.set_size_inches(12,6)
x = np.linspace(-180,180, 4320)
y = np.linspace(90,-90, 2160)

labs = ['a','b','c','d','e','f','g','h','i','j','k','l']
titles = ['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'Water regime']

bounds=[-0.2,-0.16,-0.12,-0.08,-0.04,0,0.04,0.08,0.12,0.16,0.2]


norm = colors.BoundaryNorm(bounds,10)
cmap = plt.get_cmap('PiYG',10)

xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵
for i,ax in enumerate(axs.reshape(-1)):
    m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90,ax=ax)
    if(i==0): 
        m.pcolor(xx,yy,shaps_N2O[0],cmap=cmap,latlon=True,norm=norm) 
    if(i==1): 
        m.pcolor(xx,yy,shaps_N2O[1],cmap=cmap,latlon=True,norm=norm) 
    if(i==2): 
        m.pcolor(xx,yy,shaps_N2O[2],cmap=cmap,latlon=True,norm=norm) 
        
    if(i==3): 
        m.pcolor(xx,yy,shaps_N2O[3],cmap=cmap,latlon=True,norm=norm) 
    if(i==4): 
        m.pcolor(xx,yy,shaps_N2O[4],cmap=cmap,latlon=True,norm=norm) 
    if(i==5): 
        m.pcolor(xx,yy,shaps_N2O[5],cmap=cmap,latlon=True,norm=norm) 

    if(i==6): 
        m.pcolor(xx,yy,shaps_N2O[6],cmap=cmap,latlon=True,norm=norm) 
    if(i==7): 
        m.pcolor(xx,yy,shaps_N2O[7],cmap=cmap,latlon=True,norm=norm) 
    if(i==8): 
        m.pcolor(xx,yy,shaps_N2O[8],cmap=cmap,latlon=True,norm=norm) 
    
    if(i==9): 
        m.pcolor(xx,yy,shaps_N2O[9],cmap=cmap,latlon=True,norm=norm) 
    if(i==10): 
        m.pcolor(xx,yy,shaps_N2O[10],cmap=cmap,latlon=True,norm=norm) 
    if(i==11): 
        data11=shaps_N2O[11]+shaps_N2O[12]+shaps_N2O[13]+shaps_N2O[14]
        m.pcolor(xx,yy,data11,cmap=cmap,latlon=True,norm=norm)

    m.drawcoastlines()
    m.drawcountries()
    ax.set_title(labs[i],loc='left',fontweight='bold',fontsize=15)
    ax.set_title(titles[i], loc='center')
plt.tight_layout()

cbar_ax = fig.add_axes([0.05, 0, 0.9, 0.02])
cb=matplotlib.colorbar.ColorbarBase(cbar_ax, norm=norm,ticks=bounds,cmap=cmap, orientation='horizontal',extend='neither')
cbar_ax.set_xticks(bounds)
cbar_ax.set_xticklabels(bounds)
cb.set_label("SHAP value")


# In[65]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\spatial_shap_N2O.png",dpi=600,bbox_inches='tight')


# In[66]:


###shap values绘图
fig,axs = plt.subplots(3,4)
fig.set_size_inches(12,6)
x = np.linspace(-180,180, 4320)
y = np.linspace(90,-90, 2160)

labs = ['a','b','c','d','e','f','g','h','i','j','k','l']
titles = ['Duration','MAT', 'MAP', 'Density', 'Clay', 'Nitrogen', 'TOC', 'C/N',
 'pH', 'Norg','Nin', 'Water regime']

bounds=[-3,-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4,3]


norm = colors.BoundaryNorm(bounds,10)
cmap = plt.get_cmap('PiYG',10)

xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵
for i,ax in enumerate(axs.reshape(-1)):
    m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90,ax=ax)
    if(i==0): 
        m.pcolor(xx,yy,shaps_yield[0],cmap=cmap,latlon=True,norm=norm) 
    if(i==1): 
        m.pcolor(xx,yy,shaps_yield[1],cmap=cmap,latlon=True,norm=norm) 
    if(i==2): 
        m.pcolor(xx,yy,shaps_yield[2],cmap=cmap,latlon=True,norm=norm) 
        
    if(i==3): 
        m.pcolor(xx,yy,shaps_yield[3],cmap=cmap,latlon=True,norm=norm) 
    if(i==4): 
        m.pcolor(xx,yy,shaps_yield[4],cmap=cmap,latlon=True,norm=norm) 
    if(i==5): 
        m.pcolor(xx,yy,shaps_yield[5],cmap=cmap,latlon=True,norm=norm) 

    if(i==6): 
        m.pcolor(xx,yy,shaps_yield[6],cmap=cmap,latlon=True,norm=norm) 
    if(i==7): 
        m.pcolor(xx,yy,shaps_yield[7],cmap=cmap,latlon=True,norm=norm) 
    if(i==8): 
        m.pcolor(xx,yy,shaps_yield[8],cmap=cmap,latlon=True,norm=norm) 
    
    if(i==9): 
        m.pcolor(xx,yy,shaps_yield[9],cmap=cmap,latlon=True,norm=norm) 
    if(i==10): 
        m.pcolor(xx,yy,shaps_yield[10],cmap=cmap,latlon=True,norm=norm) 
    if(i==11): 
        data11=shaps_yield[11]+shaps_yield[12]+shaps_yield[13]+shaps_yield[14]
        m.pcolor(xx,yy,data11,cmap=cmap,latlon=True,norm=norm)

    m.drawcoastlines()
    m.drawcountries()
    ax.set_title(labs[i],loc='left',fontweight='bold',fontsize=15)
    ax.set_title(titles[i], loc='center')
plt.tight_layout()

cbar_ax = fig.add_axes([0.05, 0, 0.9, 0.02])
cb=matplotlib.colorbar.ColorbarBase(cbar_ax, norm=norm,ticks=bounds,cmap=cmap, orientation='horizontal',extend='neither')
cbar_ax.set_xticks(bounds)
cbar_ax.set_xticklabels(bounds)
cb.set_label("SHAP value")


# In[67]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\spatial_shap_yield.png",dpi=600,bbox_inches='tight')


# In[136]:


rice= joblib.load('rice.pkl')


# In[58]:


data = rice[attributes]
data.to_csv('input_data.csv')
# 调整水肥措施最优化时使用

compare = rice[['CH4_xgb_per_area','N2O_xgb_per_area','yield_xgb_per_area']]
compare.to_csv('optimization/compare.csv')


# In[142]:


# 输出第二季的input_data，调整水肥措施最优化使用
attributes2 = ['duration2', 'temp', 'prec',
           'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
           'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
           'water_regime_rainfed','CH4_xgb_per_area2','N2O_xgb_per_area2','yield_xgb_per_area2']
data2 = rice[attributes2]
data2.rename(columns={'duration2': 'duration'}, inplace=True)


# In[143]:


data2


# In[9]:


# # 将提取的数据与原始的 data2 合并，并保持索引
# data2_combined = pd.concat([data2, data2_filtered]).drop_duplicates()
# # 重置索引
# data2_combined.reset_index(drop=True, inplace=True)
# # 输出合并后的数据
# print(data2_combined)


# In[144]:


# 提取 duration2 列不为 0 的行，并保持原始索引
data2_filtered = data2.loc[data2['duration'] != 0]
# 重置索引，并将原始索引添加为一列
# data2_filtered.reset_index(drop=False, inplace=True)


# In[145]:


data2_filtered


# In[148]:


compare2 = data2_filtered[['CH4_xgb_per_area2','N2O_xgb_per_area2','yield_xgb_per_area2']]
compare2.to_csv('optimization2/compare2.csv')


# In[149]:


compare2


# In[24]:


data2_filtered.to_csv('optimization2/input_data2.csv')


# In[12]:


# 假设data2是一个DataFrame对象，duration是data2中的duration列
# 使用条件筛选找出duration列大于0的行，并计算符合条件的行数
count_duration_gt_zero = data2[data2['duration'] > 0]['duration'].count()

print("Duration 列大于0的数量为:", count_duration_gt_zero)


# In[11]:


data2_filtered.to_csv('input_data2.csv')


# In[14]:


a = pd.read_csv('input_data2.csv')


# In[15]:


a


# In[ ]:





# In[41]:


np.sum(rice['yield_xgb_per_area']*rice['area'])


# In[42]:


np.sum(rice['yield_xgb'])


# In[180]:


rice[['CH4_xgb_per_area','N2O_xgb_per_area','yield_xgb_per_area']].loc[:19]


# In[ ]:





# # duration分布图

# In[1]:


filepath = "D:/li/codesdata/rice/data/duration/Rice.crop.calendar.fill.nc/duration.tif"


# In[11]:


duration = getdata(filepath,1)


# In[12]:


duration


# In[ ]:





# In[18]:


fig,ax = plt.subplots(1,1)
fig.set_size_inches(6,3)
x = np.linspace(-180,180, 4320)
y = np.linspace(90,-90, 2160)

# labs = ['a','b','c']

bounds=[80,100,120,140,160,180,200,220]
cmap = plt.get_cmap('Spectral_r',7)

min = 80
max = 220
norm1 = colors.BoundaryNorm(bounds,7)
xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵

m = Basemap(llcrnrlat=-60,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,ax=ax)
m.pcolor(xx, yy, duration, cmap=cmap, latlon=True, norm=norm1)
m.drawcoastlines()
m.drawcountries()

    # 去除边框
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

cbar_ax = fig.add_axes([0.045, 0, 0.92, 0.02])

cb = matplotlib.colorbar.ColorbarBase(cbar_ax, ticks=bounds,cmap=cmap, orientation='horizontal',norm=norm1)
cb.set_label('Duration (days)')
cb.set_ticks(bounds)
cb.set_ticklabels(bounds)
# cbar_ax.set_xticklabels(bounds)
plt.tight_layout()


# In[19]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\duration.png",dpi=600,bbox_inches='tight')

