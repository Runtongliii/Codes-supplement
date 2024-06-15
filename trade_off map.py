#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import cartopy


# In[296]:


rice=joblib.load("rice.pkl")


# In[3]:


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


# In[297]:


data=rice[['longitude','latitude','area','physical_area',
           'CH4_xgb_per_area','N2O_xgb_per_area','yield_xgb_per_area']]
data['CO2']=data['CH4_xgb_per_area']*27/1000+(data['N2O_xgb_per_area']*44*273/28)/1000
data = round(data,2)


# In[317]:


h


# In[335]:


# 计算基线权衡指数，便于后续聚类
h=pd.DataFrame(columns=['yield','GHG'])
h['yield']=(data['yield_xgb_per_area']-np.min(data['yield_xgb_per_area']))/(np.max(data['yield_xgb_per_area'])-np.min(data['yield_xgb_per_area']))
h['GHG']=(data['CO2'] - np.min(data['CO2']))/(np.max(data['CO2'])-np.min(data['CO2']))
trade_off_result = []
start = time.time()
for i in range(len(h['yield'])):

    a = np.array([h['yield'][i], h['GHG'][i]])
#     print(a)
    b = np.array([0, 0])
    c = np.array([1, 1])
    trade_off_per_grid = dist2d(a, b, c)
    
    if h['yield'][i] > h['GHG'][i]:
        comparison = 1
    elif h['yield'][i] == h['GHG'][i]:
        comparison = 0
    else:
        comparison = -1
    
    trade_off_per_grid *= comparison
    trade_off_result.append(trade_off_per_grid)
    
end = time.time()
print('Running time: %s Seconds'%(end-start))

h['tradeoff'] = trade_off_result
h['x'] = rice['longitude']
h['y'] = rice['latitude']


# In[336]:


h.describe()


# In[337]:


km = KMeans(n_clusters=3)
X=np.array(h['tradeoff']).reshape(-1,1)
km.fit(X)
print(km.cluster_centers_)

model=km
yhat = model.predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
#     # 创建这些样本的散布
#     plt.scatter(X[row_ix], X[row_ix])
#     # 绘制散点图
#     plt.show()
    print([np.min(X[where(yhat == cluster)]),np.max(X[where(yhat == cluster)])])


# In[360]:


riceh = rice.copy()
riceh['tradeoff'] = h['tradeoff']

df_1 = riceh[riceh['tradeoff'] <= -0.075]
df_2 = riceh[(riceh['tradeoff'] > -0.075) & (riceh['tradeoff'] <= 0.092)]
df_3 = riceh[riceh['tradeoff'] > 0.092]

df_1.to_csv('tradeoff/df_1.csv',index=False)
df_2.to_csv('tradeoff/df_2.csv',index=False)
df_3.to_csv('tradeoff/df_3.csv',index=False)


# In[358]:


# 输入两个数组，输出一个权衡指数数组
def tradeoff_index_per_grid(yields, GHG):
    # 计算每条数据的两个相对效益，离直线的距离为每个点的权衡系数
    # 将三维数组转换为一维数组
    yields_flat = yields.ravel()
    GHG_flat = GHG.ravel()

    ## 创建条件索引，找到非空值的索引
    valid_indices_yields = ~np.isnan(yields_flat)
    valid_indices_GHG = ~np.isnan(GHG_flat)
    
    # 使用条件索引过滤数组，得到不包含空值的数组
    yields_flat = yields_flat[valid_indices_yields]
    GHG_flat = GHG_flat[valid_indices_GHG]
    
    yieldsRB = (yields_flat - yields_flat.min()) / (yields_flat.max() - yields_flat.min())
    GHGRB = (GHG_flat - GHG_flat.min()) / (GHG_flat.max() - GHG_flat.min())
    
    
    trade_off_result = []
    start = time.time()
    for i in range(len(yieldsRB)):

        a = np.array([yieldsRB[i], GHGRB[i]])
    #     print(a)
        b = np.array([0, 0])
        c = np.array([1, 1])
        trade_off_per_grid = dist2d(a, b, c)
        
        if yieldsRB[i] > GHGRB[i]:
            comparison = 1
        elif yieldsRB[i] == GHGRB[i]:
            comparison = 0
        else:
            comparison = -1

        trade_off_per_grid *= comparison
        trade_off_result.append(trade_off_per_grid)

    end = time.time()
    print('Calculating tradeoff index: Running time: %s Seconds'%(end-start))
    
    result = pd.DataFrame(columns = ['x','y'])
    result['x'] = rice['longitude']
    result['y'] = rice['latitude']
    result['tradeoff'] = trade_off_result
    
    start=time.time()
    a = pd.merge(lonlat, result, how='left', on=['x','y'])
    tradeoff_array = np.array(a['tradeoff']).reshape(2160,4320)

    end = time.time()
    print('Convert to array: Running time: %s Seconds'%(end-start))
    
    return tradeoff_array


# In[359]:


##2081-2100 SSP126 权衡计算
future=np.load(r'D:\li\codesdata\rice\python_calculate\NEW_V2\future_prediction_npy\2081-2100ssp126.npy')
ghg=future[0]*27/1000+(future[1]*44*273/28)/1000
yields=future[2]

tradeoff_future1 = tradeoff_index_per_grid(yields, ghg)


##2081-2100 SSP585 权衡计算
future=np.load(r'D:\li\codesdata\rice\python_calculate\NEW_V2\future_prediction_npy\2081-2100ssp585.npy')
ghg=future[0]*27/1000+(future[1]*44*273/28)/1000
yields=future[2]

tradeoff_future2 = tradeoff_index_per_grid(yields, ghg)

##优化结果 权衡计算
pred_optimization = np.load(r'D:/li/codesdata/rice/python_calculate/NEW_V2/tradeoff/preds_optimization.npy')
ghg = pred_optimization[0]*27/1000+(pred_optimization[1]*44*273/28)/1000
yields = pred_optimization[2]

tradeoff_optimization = tradeoff_index_per_grid(yields,ghg)

##基线权衡计算
baseline_prediction = np.load(r"D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/global_prediction.npy")
ghg = baseline_prediction[0]*27/1000+(baseline_prediction[1]*44*273/28)/1000
yields = baseline_prediction[2]

tradeoff_baseline = tradeoff_index_per_grid(yields,ghg)


# In[ ]:


# ###计算权衡程度三种类型面积比例
# def getdata(filepath,band):
#     dataset=rasterio.open(filepath)
#     data = dataset.read(band)
#     data[data<-9999]=np.nan
#     return data

# areapath = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"
# area=getdata(areapath,1)
# area=np.where(area<0,0,area)
# low_indices = np.where((tradeoff_spatial >= -0.8) & (tradeoff_spatial < -0.26))
# medium_indices = np.where((tradeoff_spatial >= -0.26) & (tradeoff_spatial < 0.001))
# high_indices = np.where((tradeoff_spatial >= 0.001) & (tradeoff_spatial <= 0.84))
# # 计算每个分类下area的和
# frac11 = np.sum(area[low_indices])/np.nansum(area)
# frac12 = np.sum(area[medium_indices])/np.nansum(area)
# frac13 = np.sum(area[high_indices])/np.nansum(area)

# low_indices = np.where((tradeoff_future1 >= -0.8) & (tradeoff_future1 < -0.26))
# medium_indices = np.where((tradeoff_future1 >= -0.26) & (tradeoff_future1 < 0.001))
# high_indices = np.where((tradeoff_future1 >= 0.001) & (tradeoff_future1 <= 0.84))
# # 计算每个分类下area的和
# frac21 = np.sum(area[low_indices])/np.nansum(area)
# frac22 = np.sum(area[medium_indices])/np.nansum(area)
# frac23 = np.sum(area[high_indices])/np.nansum(area)

# low_indices = np.where((tradeoff_future2 >= -0.8) & (tradeoff_future2 < -0.26))
# medium_indices = np.where((tradeoff_future2 >= -0.26) & (tradeoff_future2 < 0.001))
# high_indices = np.where((tradeoff_future2 >= 0.001) & (tradeoff_future2 <= 0.84))
# # 计算每个分类下area的和
# frac31 = np.sum(area[low_indices])/np.nansum(area)
# frac32 = np.sum(area[medium_indices])/np.nansum(area)
# frac33 = np.sum(area[high_indices])/np.nansum(area)


# print([frac11,frac12,frac13,frac21,frac22,frac23,frac31,frac32,frac33])


# In[354]:


# areapath = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"
# area=getdata(areapath,1)
# area=np.where(area<0,0,area)
# low_indices = np.where((trade_off_optimization >= -0.48) & (trade_off_optimization < -0.075))
# medium_indices = np.where((trade_off_optimization >= -0.075) & (trade_off_optimization < 0.092))
# high_indices = np.where((trade_off_optimization >= 0.092) & (trade_off_optimization <= 0.48))
# # 计算每个分类下area的和
# frac11 = np.sum(area[low_indices])/np.nansum(area)
# frac12 = np.sum(area[medium_indices])/np.nansum(area)
# frac13 = np.sum(area[high_indices])/np.nansum(area)

# print(frac11,frac12,frac13)


# In[308]:


# h=pd.DataFrame(columns=['yield','GHG'])
# h['yield']=(data['yield_xgb_per_area']-np.min(data['yield_xgb_per_area']))/(np.max(data['yield_xgb_per_area'])-np.min(data['yield_xgb_per_area']))
# h['GHG']=(np.max(data['CO2'])-data['CO2'])/(np.max(data['CO2'])-np.min(data['CO2']))
# h['tradeoff']=h['yield']-h['GHG']
# h['x']=rice['longitude']
# h['y']=rice['latitude']


# In[10]:


from sklearn.cluster import KMeans
from numpy import unique
from numpy import where


# In[16]:


# attributes=['duration', 'temp', 'prec',
#        'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
#        'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
#        'water_regime_rainfed']
# X=rice[attributes]
# distance = []
# k = []
# for n_clusters in range(1,10):
#     cls = KMeans(n_clusters).fit(X)

#     #曼哈顿距离
#     def manhattan_distance(x,y):
#         return np.sum(abs(x-y))

#     distance_sum = 0
#     for i in range(n_clusters):
#         group = cls.labels_ == i
#         members = X[group,:]
#         for v in members:
#             distance_sum += manhattan_distance(np.array(v), cls.cluster_centers_[i])
#     distance.append(distance_sum)
#     k.append(n_clusters)
# plt.scatter(k, distance)
# plt.plot(k, distance)
# plt.xlabel("k")
# plt.ylabel("distance")
# plt.show()


# In[28]:


# 导入经纬度
lonlat = pd.read_csv(r"D:\li\codesdata\rice\data\lonlat.csv")
lonlat=round(lonlat,2)


# In[60]:


# pred_optimization = np.load(r'D:/li/codesdata/rice/python_calculate/NEW_V2/tradeoff/preds_optimization.npy')
# ghg=pred_optimization[0]*27/1000+(pred_optimization[1]*44*273/28)/1000
# yields=pred_optimization[2]
# ghg_benefit=(np.nanmax(ghg)-ghg)/(np.nanmax(ghg)-np.nanmin(ghg))
# yields_benefit = (yields-np.nanmin(yields))/(np.nanmax(yields)-np.nanmin(yields))
# trade_off_optimization = yields_benefit-ghg_benefit


# areapath = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"
# area=getdata(areapath,1)
# area=np.where(area<0,0,area)
# low_indices = np.where((trade_off_optimization >= -0.8) & (trade_off_optimization < -0.26))
# medium_indices = np.where((trade_off_optimization >= -0.26) & (trade_off_optimization < 0.001))
# high_indices = np.where((trade_off_optimization >= 0.001) & (trade_off_optimization <= 0.84))
# # 计算每个分类下area的和
# frac11 = np.sum(area[low_indices])/np.nansum(area)
# frac12 = np.sum(area[medium_indices])/np.nansum(area)
# frac13 = np.sum(area[high_indices])/np.nansum(area)

# print(frac11,frac12,frac13)


# In[30]:


start=time.time()
a=pd.merge(lonlat,h,how='left',on=['x','y'])
tradeoff_spatial=np.array(a['tradeoff']).reshape(2160,4320)

end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[366]:


np.nanmin(tradeoff_future1)


# In[367]:


np.nanmin(tradeoff_future2)


# In[368]:


###计算权衡程度三种类型面积比例
def getdata(filepath,band):
    dataset=rasterio.open(filepath)
    data = dataset.read(band)
    data[data<-9999]=np.nan
    return data

areapath = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"
area=getdata(areapath,1)
area=np.where(area<0,0,area)
low_indices = np.where(tradeoff_baseline < -0.075)
medium_indices = np.where((tradeoff_baseline >= -0.075) & (tradeoff_baseline < 0.092))
high_indices = np.where(tradeoff_baseline >= 0.092)
# 计算每个分类下area的和
frac11 = np.sum(area[low_indices])/np.nansum(area)
frac12 = np.sum(area[medium_indices])/np.nansum(area)
frac13 = np.sum(area[high_indices])/np.nansum(area)


low_indices = np.where(tradeoff_optimization < -0.075)
medium_indices = np.where((tradeoff_optimization >= -0.075) & (tradeoff_optimization < 0.092))
high_indices = np.where(tradeoff_optimization >= 0.092)
# 计算每个分类下area的和
frac21 = np.sum(area[low_indices])/np.nansum(area)
frac22 = np.sum(area[medium_indices])/np.nansum(area)
frac23 = np.sum(area[high_indices])/np.nansum(area)


low_indices = np.where(tradeoff_future1 < -0.075)
medium_indices = np.where((tradeoff_future1 >= -0.075) & (tradeoff_future1 < 0.092))
high_indices = np.where(tradeoff_future1 >= 0.092)
# 计算每个分类下area的和
frac31 = np.sum(area[low_indices])/np.nansum(area)
frac32 = np.sum(area[medium_indices])/np.nansum(area)
frac33 = np.sum(area[high_indices])/np.nansum(area)

low_indices = np.where(tradeoff_future2 < -0.075)
medium_indices = np.where((tradeoff_future2 >= -0.075) & (tradeoff_future2 < 0.092))
high_indices = np.where(tradeoff_future2 >= 0.092)
# 计算每个分类下area的和
frac41 = np.sum(area[low_indices])/np.nansum(area)
frac42 = np.sum(area[medium_indices])/np.nansum(area)
frac43 = np.sum(area[high_indices])/np.nansum(area)

print([frac11,frac12,frac13,frac21,frac22,frac23,frac31,frac32,frac33,frac41,frac42,frac43])


# In[375]:


from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np



fig, axs = plt.subplots(2, 2)
fig.set_size_inches(8, 5)  # 调整图形大小
x = np.linspace(-180, 180, tradeoff_spatial.shape[1])
y = np.linspace(90, -90, tradeoff_spatial.shape[0])

spatial_tradeoffs = [tradeoff_baseline, tradeoff_optimization, tradeoff_future1, tradeoff_future2]
labs = ['a', 'b', 'c', 'd']  # 更新标签数
cols = ['#8dd3c7', '#ffffb3', '#bebada']
cmap = colors.ListedColormap(cols)

bounds = [-1, -0.075, 0.092, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)
xx, yy = np.meshgrid(x, y)

for i, ax in enumerate(axs.reshape(-1)):
    m = Basemap(llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax)
    m.pcolor(xx, yy, spatial_tradeoffs[i], cmap=cmap, latlon=True, norm=norm)
    m.drawcoastlines()
    m.drawcountries()
    ax.set_title(labs[i], loc='left', fontweight='bold', fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(bottom=False, left=False)  # 关闭刻度线
    ax.tick_params(labelbottom=False, labelleft=False)  # 关闭刻度标签
    ax.set_yticks([])  # 移除y轴刻度

# 绘制第一个子图的柱状图
ax1 = plt.axes([0.08, 0.53, 0.02, 0.18])  # 调整位置
ax1.set_yticks([0, 1])
ax1.set_yticklabels([0, 1], fontsize=10)
ax1.set_ylabel('Area fraction', fontsize=10)
ax1.get_xaxis().set_ticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.bar(x=[0], height=frac11, width=0.05, color=cols[0], edgecolor='black')
plt.bar(x=[0], height=frac12, width=0.05, color=cols[1], bottom=frac11, edgecolor='black')
plt.bar(x=[0], height=frac13, width=0.05, color=cols[2], bottom=frac11 + frac12, edgecolor='black')
plt.text(0.03, 0.0, '17.6%', color='black', fontsize=10)
plt.text(0.03, 0.4, '58.7%', color='black', fontsize=10)
plt.text(0.03, 0.82, '23.7%', color='black', fontsize=10)

# 绘制第二个子图的柱状图
ax2 = plt.axes([0.57, 0.53, 0.02, 0.18])  # 调整位置
ax2.set_yticks([0, 1])
ax2.set_yticklabels([0, 1], fontsize=10)
ax2.get_xaxis().set_ticks([])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.bar(x=[0], height=frac21, width=0.05, color=cols[0], edgecolor='black')
plt.bar(x=[0], height=frac22, width=0.05, color=cols[1], bottom=frac21, edgecolor='black')
plt.bar(x=[0], height=frac23, width=0.05, color=cols[2], bottom=frac21 + frac22, edgecolor='black')
plt.text(0.03, 0.0, '14.4%', color='black', fontsize=10)
plt.text(0.03, 0.44, '63.2%', color='black', fontsize=10)
plt.text(0.03, 0.82, '22.4%', color='black', fontsize=10)

# 绘制第三个子图的柱状图
ax3 = plt.axes([0.08, 0.08, 0.02, 0.18])  # 调整位置
ax3.set_yticks([0, 1])
ax3.set_yticklabels([0, 1], fontsize=10)
ax3.set_ylabel('Area fraction', fontsize=10)
ax3.get_xaxis().set_ticks([])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plt.bar(x=[0], height=frac31, width=0.05, color=cols[0], edgecolor='black')
plt.bar(x=[0], height=frac32, width=0.05, color=cols[1], bottom=frac31, edgecolor='black')
plt.bar(x=[0], height=frac33, width=0.05, color=cols[2], bottom=frac31 + frac32, edgecolor='black')
plt.text(0.03, 0.0, '19.9%', color='black', fontsize=10)
plt.text(0.03, 0.42, '55.4%', color='black', fontsize=10)
plt.text(0.03, 0.82, '24.7%', color='black', fontsize=10)

# 绘制第四个子图的柱状图
ax4 = plt.axes([0.57, 0.08, 0.02, 0.18])  # 调整位置
ax4.set_yticks([0, 1])
ax4.set_yticklabels([0, 1], fontsize=10)
ax4.get_xaxis().set_ticks([])
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plt.bar(x=[0], height=frac41, width=0.05, color=cols[0], edgecolor='black')
plt.bar(x=[0], height=frac42, width=0.05, color=cols[1], bottom=frac41, edgecolor='black')
plt.bar(x=[0], height=frac43, width=0.05, color=cols[2], bottom=frac41 + frac42, edgecolor='black')
plt.text(0.03, 0.0, '18.7%', color='black', fontsize=10)
plt.text(0.03, 0.4, '47.5%', color='black', fontsize=10)
plt.text(0.03, 0.8, '33.8%', color ='black', fontsize=10)

cbar_ax = fig.add_axes([0.11, 0, 0.79, 0.02])
labels=['High emission and low yield','Balanced trade-off','High yield and low emission']
ticks=[(bounds[i]+bounds[i+1])/2 for i in range(0,3)]
cb1=matplotlib.colorbar.ColorbarBase(cbar_ax, norm=norm,ticks=ticks,cmap=cmap, orientation='horizontal')
cbar_ax.set_xticks(ticks)
cbar_ax.set_xticklabels(labels,fontsize=10)



plt.tight_layout()
plt.show()


# In[376]:


fig.savefig("E:/li/文章/审稿人意见/李润桐-NF-0810/figs/tradeoff_category.png",bbox_inches='tight',dpi=600)


# In[41]:


# #####2080-2100 SSP126权衡
# from matplotlib import colors
# fig,axs = plt.subplots(3,1)
# fig.set_size_inches(6,10)
# x = np.linspace(-180,180, tradeoff_spatial.shape[1])
# y = np.linspace(90,-90, tradeoff_spatial.shape[0])

# spatial_tradeoffs=[tradeoff_baseline, tradeoff_optimization, tradeoff_future1, tradeoff_future2]
# labs = ['a','b','c','d']
# cols=['#8dd3c7','#ffffb3','#bebada']
# cmap = colors.ListedColormap(cols)


# bounds=[-1,-0.075,0.092,1]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵


# for i,ax in enumerate(axs.reshape(-1)):
#     m = Basemap(llcrnrlat=-60,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,ax=ax)
#     m.pcolor(xx,yy,spatial_tradeoffs[i],cmap=cmap,latlon=True,norm=norm) 
#     m.drawcoastlines()
#     m.drawcountries()
#     ax.set_title(labs[i],loc='left',fontweight='bold',fontsize = 20)
#         # 去除边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
# plt.tight_layout()




# ###绘制三种分类面积比例柱状图
# ax1=plt.axes([0.08, 0.71, 0.02, 0.08])
# ax1.set_yticks([0,1])
# ax1.set_yticklabels([0,1], fontsize=10)
# ax1.set_ylabel('Area fraction', fontsize=10)
# ax1.get_xaxis().set_ticks([])
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# plt.bar(x=[0],height=frac11,width=0.05,color=cols[0],edgecolor='black')
# plt.bar(x=[0], height=frac12, width=0.05, color=cols[1], bottom=frac11,edgecolor='black')
# plt.bar(x=[0], height=frac13, width=0.05, color=cols[2], bottom=frac11+frac12,edgecolor='black')
# plt.text(0.03, 0.0, '11%', color='black',fontsize=10)
# plt.text(0.03, 0.35, '54%', color='black',fontsize=10)
# plt.text(0.03, 0.75, '35%', color='black',fontsize=10)


# ###绘制三种分类面积比例柱状图
# ax2=plt.axes([0.08, 0.4, 0.02, 0.08])
# ax2.set_yticks([0,1])
# ax2.set_yticklabels([0,1], fontsize=10)
# ax2.set_ylabel('Area fraction', fontsize=10)
# ax2.get_xaxis().set_ticks([])
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# plt.bar(x=[0],height=frac21,width=0.05,color=cols[0],edgecolor='black')
# plt.bar(x=[0], height=frac22, width=0.05, color=cols[1], bottom=frac21,edgecolor='black')
# plt.bar(x=[0], height=frac23, width=0.05, color=cols[2], bottom=frac21+frac22,edgecolor='black')
# plt.text(0.03, 0.0, '13%', color='black',fontsize=10)
# plt.text(0.03, 0.35, '57%', color='black',fontsize=10)
# plt.text(0.03, 0.75, '30%', color='black',fontsize=10)


# ###绘制三种分类面积比例柱状图
# ax3=plt.axes([0.08, 0.09, 0.02, 0.08])
# ax3.set_yticks([0,1])
# ax3.set_yticklabels([0,1], fontsize=10)
# ax3.set_ylabel('Area fraction', fontsize=10)
# ax3.get_xaxis().set_ticks([])
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# plt.bar(x=[0],height=frac31,width=0.05,color=cols[0],edgecolor='black')
# plt.bar(x=[0], height=frac32, width=0.05, color=cols[1], bottom=frac31,edgecolor='black')
# plt.bar(x=[0], height=frac33, width=0.05, color=cols[2], bottom=frac31+frac32,edgecolor='black')
# plt.text(0.03, 0.0, '7%', color='black',fontsize=10)
# plt.text(0.03, 0.3, '57%', color='black',fontsize=10)
# plt.text(0.03, 0.75, '36%', color='black',fontsize=10)





# cbar_ax = fig.add_axes([0.11, 0, 0.79, 0.02])
# labels=['Greenhouse gas priority','Balanced trade-off','Yield priority']
# ticks=[(bounds[i]+bounds[i+1])/2 for i in range(0,3)]
# cb1=matplotlib.colorbar.ColorbarBase(cbar_ax, norm=norm,ticks=ticks,cmap=cmap, orientation='horizontal')
# cbar_ax.set_xticks(ticks)
# cbar_ax.set_xticklabels(labels,fontsize=10)

# plt.tight_layout()


# In[43]:


fig.savefig("E:/li/文章/审稿人意见/李润桐-NF-0810/figs/tradeoff.png",bbox_inches='tight',dpi=600)


# In[ ]:





# # 计算权衡指数-输入数组，输出RMSE数对和权衡指数，使用这个权衡指数

# In[68]:


np.nanmean(ghg_benefit)


# In[69]:


pred_optimization = np.load(r'D:/li/codesdata/rice/python_calculate/NEW_V2/tradeoff/preds_optimization.npy')
GHG = pred_optimization[0]*27/1000+(pred_optimization[1]*44*273/28)/1000
yields = pred_optimization[2]


# In[70]:




# 将三维数组转换为一维数组
yields_flat = yields.ravel()
GHG_flat = GHG.ravel()

## 创建条件索引，找到非空值的索引
valid_indices_yields = ~np.isnan(yields_flat)
valid_indices_GHG = ~np.isnan(GHG_flat)

# 使用条件索引过滤数组，得到不包含空值的数组
yields_flat = yields_flat[valid_indices_yields]
GHG_flat = GHG_flat[valid_indices_GHG]


# In[74]:


yieldsRB = (yields_flat.max()-yields_flat ) / (yields_flat.max() - yields_flat.min())
GHGRB = (GHG_flat - GHG_flat.min()) / (GHG_flat.max() - GHG_flat.min())

# 计算地上和地下相对效益的均值
yieldsmean = yieldsRB.mean()
GHGmean = GHGRB.mean()


# In[76]:


yieldsRMSE = np.sqrt(((yieldsRB - yieldsmean) ** 2).sum() / (len(yieldsRB) - 1)) #yield
GHGRMSE = np.sqrt(((GHGRB - GHGmean) ** 2).sum() / (len(GHGRB) - 1)) #GHG


# In[80]:


# 计算点到直线的距离
a = np.array([yieldsRMSE, GHGRMSE])
b = np.array([0, 0])
c = np.array([1, 1])
tradeoff_index = dist2d(a, b, c)
tradeoff_index


# In[151]:


yields_RMSE


# In[169]:


# 横轴yield，纵轴GHG
def calculate_tradeoff_index(yields, GHG):
    
    # 将三维数组转换为一维数组
    yields_flat = yields.ravel()
    GHG_flat = GHG.ravel()

    ## 创建条件索引，找到非空值的索引
    valid_indices_yields = ~np.isnan(yields_flat)
    valid_indices_GHG = ~np.isnan(GHG_flat)

    # 使用条件索引过滤数组，得到不包含空值的数组
    yields_flat = yields_flat[valid_indices_yields]
    GHG_flat = GHG_flat[valid_indices_GHG]

    yieldsRB = (yields_flat - yields_flat.min()) / (yields_flat.max() - yields_flat.min())
    GHGRB = (GHG_flat - GHG_flat.min()) / (GHG_flat.max() - GHG_flat.min())

    # 计算产量和GHG相对效益的均值
    yieldsmean = yieldsRB.mean()
    GHGmean = GHGRB.mean()

    # 计算产量和GHG的均方根误差(RMSE)
    yieldsRMSE = np.sqrt(((yieldsRB - yieldsmean) ** 2).sum() / (len(yieldsRB) - 1)) #yield
    GHGRMSE = np.sqrt(((GHGRB - GHGmean) ** 2).sum() / (len(GHGRB) - 1)) #GHG

    # 计算点到直线的距离
    a = np.array([yieldsRMSE, GHGRMSE])
    b = np.array([0, 0])
    c = np.array([1, 1])
    tradeoff_index = dist2d(a, b, c)
#     # 根据 rmse[1] 和 rmse[2] 的大小关系确定正负号
#     print(yieldsRMSE)
#     print(GHGRMSE)
    
    if yieldsRMSE > GHGRMSE:
        comparison = 1
    elif yieldsRMSE == GHGRMSE:
        comparison = 0
    else:
        comparison = -1
    
    tradeoff_index *= comparison
    # 高产低排放为正，低产高排放为负数，数值越大越有利
#     print(comparison)
#     print(tradeoff_index)

    # 返回均方根误差和权衡指数
    return yieldsRMSE, GHGRMSE, tradeoff_index

# 计算点到直线的距离
def dist2d(a, b, c):
    v1 = b - c
    v2 = a - b
    m = np.vstack([v1, v2])
    d = np.abs(np.linalg.det(m)) / np.sqrt((v1 ** 2).sum())
    return d


# In[ ]:





# In[165]:


Baseline_prediction_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/global_prediction.npy"
Optimization_prediction_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/tradeoff/preds_optimization.npy"

future_2021_2040ssp126_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2021-2040ssp126.npy"
future_2021_2040ssp585_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2021-2040ssp585.npy"

future_2041_2060ssp126_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2041-2060ssp126.npy"
future_2041_2060ssp585_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2041-2060ssp585.npy"

future_2061_2080ssp126_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2061-2080ssp126.npy"
future_2061_2080ssp585_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2061-2080ssp585.npy"

future_2081_2100ssp126_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2081-2100ssp126.npy"
future_2081_2100ssp585_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2081-2100ssp585.npy"


# In[166]:


def trade_off_index_per_scenario(data):
    # 读取未来预测数据
#     data = np.load(filepath)
    # 计算GHG和yields
    ghg = data[0] * 27 / 1000 + (data[1] * 44 * 273 / 28) / 1000
    yields = data[2]

    # 计算权衡指数
    yields_RMSE, GHG_RMSE, tradeoff_index = calculate_tradeoff_index(yields, ghg)

    # 创建数据框保存结果
    result_df = pd.DataFrame({
        'Yields_RMSE': [yields_RMSE],
        'GHG_RMSE': [GHG_RMSE],
        'Tradeoff_index': [tradeoff_index]
    })
    return result_df


# In[174]:


file_paths


# In[172]:


import pandas as pd

# 文件路径列表
file_paths = [
    Baseline_prediction_path,
    Optimization_prediction_path,
    future_2021_2040ssp126_path,
    future_2021_2040ssp585_path,
    future_2041_2060ssp126_path,
    future_2041_2060ssp585_path,
    future_2061_2080ssp126_path,
    future_2061_2080ssp585_path,
    future_2081_2100ssp126_path,
    future_2081_2100ssp585_path
]

# 存储结果的列表
result_dfs = []

# 循环处理每个文件路径
for i in range(len(file_paths)):
    file_path = file_paths[i]
    data = np.load(file_path)
    
    # 调用函数计算权衡指数
    result_df = trade_off_index_per_scenario(data)
    
    # 提取文件名作为列名
    file_name = file_path.split('/')[-1].split('.')[0]
    # 添加文件名列
    result_df['File'] = file_name
    # 将结果添加到结果列表中
    result_dfs.append(result_df)

# 合并所有结果
final_result_df = pd.concat(result_dfs, ignore_index=True)

# 保存结果为CSV文件
# final_result_df.to_csv('tradeoff_results_all.csv', index=False)


# In[178]:


final_result_df['File']


# In[187]:


final_result_df


# In[289]:


# 绘制优化前后的权衡指数散点
import matplotlib.pyplot as plt

# 提取数据
Yields_RMSE = final_result_df['Yields_RMSE'][:2]
GHG_RMSE = final_result_df['GHG_RMSE'][:2]
File = ['Baseline prediction','Optimization prediction']
Tradeoff_index = final_result_df['Tradeoff_index'][:2]

# 定义颜色映射
color_map = {'Baseline prediction': 'blue', 'Optimization prediction': 'orange'}
#  设置图的大小，使其横纵相同
plt.figure(figsize=(4, 4))
# 绘制散点图
for i in range(2):
    plt.scatter(Yields_RMSE[i], GHG_RMSE[i], color=color_map[File[i]], label=File[i])

## 添加 y=x 这条线
plt.plot([0.10, 0.15], [0.10, 0.15], color='black', linestyle='--', label='1:1 line')

# 设置横纵轴范围相同
plt.axis([0.10, 0.15, 0.10, 0.15])

# 添加标题和标签
# plt.title('Scatter Plot of Yields_RMSE and GHG_RMSE')
plt.xlabel('Overall benefit for Yields')
plt.ylabel('Overall benefit for GHG')

# 添加颜色图例，并将图例放在图内边框
legend = plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.99), borderaxespad=0)

# 去除图例边缘线
legend.get_frame().set_linewidth(0)

# 计算直线的斜率和截距
slope = 1
intercept = 0

# 绘制直线

# 添加箭头
for i in range(2):
    # 计算直线上对应点的x坐标
    x = (Yields_RMSE[i] + slope * GHG_RMSE[i] - slope * intercept) / (slope ** 2 + 1)
    # 计算直线上对应点的y坐标
    y = slope * x + intercept
    # 添加箭头
    plt.annotate('', xy=(Yields_RMSE[i], GHG_RMSE[i]), xytext=(x, y), 
                 arrowprops=dict(arrowstyle='->', color='black'))

# 去除图后的网格
plt.grid(False)
plt.savefig(r'E:\li\文章\审稿人意见\李润桐-NF-0810\figs\trade_off_illustration.png', dpi=600, bbox_inches='tight')
plt.savefig(r'E:\li\文章\审稿人意见\李润桐-NF-0810\figs\trade_off_illustration.svg', dpi=600, bbox_inches='tight')
# 显示图形
plt.show()


# In[293]:


# 定义每个分组的数据和颜色
group_data = {
    'Baseline': {'Data': final_result_df['Tradeoff_index'][:2], 'Color': ['blue', 'orange']},
    '2021-2040': {'Data': final_result_df['Tradeoff_index'][2:4], 'Color': 'black'},
    '2041-2060': {'Data': final_result_df['Tradeoff_index'][4:6], 'Color': 'black'},
    '2061-2080': {'Data': final_result_df['Tradeoff_index'][6:8], 'Color': 'black'},
    '2081-2100': {'Data': final_result_df['Tradeoff_index'][8:], 'Color': 'black'}
}

# 设置图的大小
plt.figure(figsize=(5, 4))

# 绘制散点图和折线
for label, data in group_data.items():
    if label == 'Baseline':
        plt.scatter([label] * len(data['Data']), data['Data'], label=label, color=data['Color'])
    else:
        plt.scatter([label] * len(data['Data']), data['Data'], label=label, color='black')

# 添加折线
plt.plot(['2021-2040', '2041-2060', '2061-2080', '2081-2100'], 
         [group_data['2021-2040']['Data'][2], group_data['2041-2060']['Data'][4], 
          group_data['2061-2080']['Data'][6], group_data['2081-2100']['Data'][8]], color='purple', linestyle='-', marker='')
plt.plot(['2021-2040', '2041-2060', '2061-2080', '2081-2100'], 
         [group_data['2021-2040']['Data'][3], group_data['2041-2060']['Data'][5], 
          group_data['2061-2080']['Data'][7], group_data['2081-2100']['Data'][9]], color='red', linestyle='-', marker='')

# 添加箭头
plt.annotate('Baseline', xy=('Baseline', group_data['Baseline']['Data'][0]), xytext=(30, 0), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'))
plt.annotate('Optimization', xy=('Baseline', group_data['Baseline']['Data'][1]), xytext=(30, 0), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'))



# 添加标题和标签
# plt.title('Tradeoff Index')
plt.xlabel('')
plt.ylabel('Tradeoff Index')

# 添加网格线
plt.grid(False)

# 添加图例
plt.legend(['SSP126', 'SSP585'])

# 显示图形
plt.show()


# In[272]:


Yields_RMSE


# In[294]:


# 创建具有两个子图的图形


# 创建具有两个子图的图形，其中第一个子图占据整个图的40％，第二个子图占据60％
fig = plt.figure(figsize=(10, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[4,5])

# 创建子图
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# 提取数据
Yields_RMSE = final_result_df['Yields_RMSE'][:2]
GHG_RMSE = final_result_df['GHG_RMSE'][:2]
File = ['Baseline prediction','Optimization prediction']
Tradeoff_index = final_result_df['Tradeoff_index'][:2]

# 定义颜色映射
color_map = {'Baseline prediction': 'blue', 'Optimization prediction': 'orange'}


# 第一个子图：散点图
for i in range(2):
    ax1.scatter(Yields_RMSE[i], GHG_RMSE[i], color=color_map[File[i]], label=File[i])
ax1.plot([0.10, 0.15], [0.10, 0.15], color='black', linestyle='--', label='1:1 line')
ax1.axis([0.10, 0.15, 0.10, 0.15])
ax1.set_xlabel('Overall benefit for Yields')
ax1.set_ylabel('Overall benefit for GHG')
legend = ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.99), borderaxespad=0)
# 去除图例边缘线
legend.get_frame().set_linewidth(0)
ax1.grid(False)
# 计算直线的斜率和截距
slope = 1
intercept = 0

# 绘制直线

# 添加箭头
for i in range(2):
    # 计算直线上对应点的x坐标
    x = (Yields_RMSE[i] + slope * GHG_RMSE[i] - slope * intercept) / (slope ** 2 + 1)
    # 计算直线上对应点的y坐标
    y = slope * x + intercept
    # 添加箭头
    ax1.annotate('', xy=(Yields_RMSE[i], GHG_RMSE[i]), xytext=(x, y), 
                 arrowprops=dict(arrowstyle='->', color='black'))


# 在第二个子图上绘制第二个散点图和箭头
for label, data in group_data.items():
    if label == 'Baseline':
        ax2.scatter([label] * len(data['Data']), data['Data'], label=label, color=data['Color'])
    else:
        ax2.scatter([label] * len(data['Data']), data['Data'], label=label, color='black')
ax2.plot(['2021-2040', '2041-2060', '2061-2080', '2081-2100'], 
         [group_data['2021-2040']['Data'][2], group_data['2041-2060']['Data'][4], 
          group_data['2061-2080']['Data'][6], group_data['2081-2100']['Data'][8]], color='purple', linestyle='-', marker='')
ax2.plot(['2021-2040', '2041-2060', '2061-2080', '2081-2100'], 
         [group_data['2021-2040']['Data'][3], group_data['2041-2060']['Data'][5], 
          group_data['2061-2080']['Data'][7], group_data['2081-2100']['Data'][9]], color='red', linestyle='-', marker='')
ax2.set_xlabel('Group')
ax2.set_ylabel('Tradeoff Index')
ax2.grid(False)
ax2.legend(['SSP126', 'SSP585'])
ax2.annotate('Baseline', xy=('Baseline', group_data['Baseline']['Data'][0]), xytext=(30, 0), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'))
ax2.annotate('Optimization', xy=('Baseline', group_data['Baseline']['Data'][1]), xytext=(30, 0), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'))
ax1.set_title("a",loc='left',fontweight='bold',fontsize=20)
ax2.set_title("b",loc='left',fontweight='bold',fontsize=20)

# 调整子图之间的间距
plt.tight_layout()

# 保存图形
plt.savefig(r'E:\li\文章\审稿人意见\李润桐-NF-0810\figs\trade_off_scatter_combined.png', dpi=600, bbox_inches='tight')

# 显示图形
plt.show()

