#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
#python中赋值，如果是可变对象，对其中一个修改会影响到另一个。如果要生成完全新的对象，应使用deepcopy
import joblib
import time
import os


# In[159]:


files=os.listdir('D:/li/temporary/future_climate_extract/')
#print(type(files))
print(files)
paths=[]
for i in files:
    path=os.path.join('D:/li/temporary/future_climate_extract/'+i)
    paths.append(path)


# In[3]:


attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']


# In[8]:


rice=joblib.load('rice.pkl')
scaler_CH4 = joblib.load('scaler_CH4.pkl')
scaler_N2O = joblib.load('scaler_N2O.pkl')
scaler_yield = joblib.load('scaler_yield.pkl')
xgbr_CH4=joblib.load("xgbr_CH4.pkl")
xgbr_N2O=joblib.load("xgbr_N2O.pkl")
xgbr_yield=joblib.load("xgbr_yield.pkl")


# In[5]:


models=['CNRM-CM6-1','CanESM5','CNRM-CM6-1-HR','EC-Earth3-Veg','GISS-E2-1-G','GISS-E2-1-H',
        'HadGEM3-GC31-LL','INM-CM4-8','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-HR','UKESM1-0-LL']


# In[9]:


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

    return pred


# In[10]:


attributes=['duration', 'temp', 'prec',
           'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
           'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
           'water_regime_rainfed']
attributes2 = ['duration2', 'temp', 'prec',
           'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
           'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
           'water_regime_rainfed']


# In[12]:


def future_scenario_prediction(paths,i):
    temp_rice=pd.read_csv(os.path.join(paths[i]+'/temp_rice.csv'))
    prec_rice=pd.read_csv(os.path.join(paths[i]+'/prec_rice.csv'))
    
    print(str(i)+" 气候降水数据读取")
    #修改列名
    newcolumns=['longitude', 'latitude']
    for c in range(2,len(temp_rice.columns)):
        #print(temp_rice.columns[i])
        n=temp_rice.columns[c][14:-21]  ##选择出模式名
        newcolumns.append(n)
    temp_rice.columns=newcolumns
    prec_rice.columns=newcolumns
    
    #补充缺失值
    #上下两个值的平均值进行填充
    for data in [temp_rice,prec_rice]:
        data.fillna(data.interpolate(),inplace=True)
        
    print("缺失值填充")
    start = time.time()
    rice=joblib.load("rice.pkl")
    
    rice=rice.iloc[:,0:30]
    
    rice_CH4=[]
    rice_N2O=[]
    rice_yield=[]
    
    for model in models:
        rice["temp"]=temp_rice[model]
        rice["prec"]=prec_rice[model]
        #name=names[k]
        rice_CH4.append(rice_cal(rice,xgbr_CH4,scaler_CH4,attributes))
        rice_N2O.append(rice_cal(rice,xgbr_N2O,scaler_N2O,attributes))
        rice_yield.append(rice_cal(rice,xgbr_yield,scaler_yield,attributes))
    end = time.time()
    print(' Running time: %s Seconds'%(end-start))
    
    
    
    file_names=["rice_CH4","rice_N2O","rice_yield"]
    out_data=[rice_CH4,rice_N2O,rice_yield]
    #names=temp_rice.columns[2:temp_single_rice.shape[1]+2]
    start =time.time()
    for j in range(3):
        data=out_data[j]
        data=pd.DataFrame(data).T
        data.columns=models
        data['x']=rice['longitude']
        data['y']=rice['latitude']
        data['area']=rice['area']
        data.to_csv(os.path.join(paths[i]+'/pred/'+file_names[j]+".csv"))
    end = time.time()
    print('Save Running time: %s Seconds'%(end-start))


# In[14]:


for i in range(0,8):
    future_scenario_prediction(paths,i)


# In[40]:


import pandas as pd

# 禁止显示 SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # 默认为'warn'

# 在这之后的代码将不会显示 SettingWithCopyWarning
lonlat = pd.read_csv(r"D:\li\codesdata\rice\data\lonlat.csv")
lonlat=round(lonlat,2)


# In[273]:


###转npy文件
def future2npy(i):
    rice_CH4=pd.read_csv(os.path.join(paths[i]+'/pred/rice_CH4.csv'))
    rice_N2O=pd.read_csv(os.path.join(paths[i]+'/pred/rice_N2O.csv'))
    rice_yield=pd.read_csv(os.path.join(paths[i]+'/pred/rice_yield.csv'))
    results=rice_CH4[['x','y']]
    
    
    results.loc[:, 'CH4'] = np.average(rice_CH4[models], axis=1)
    results.loc[:, 'N2O'] = np.average(rice_N2O[models], axis=1)
    results.loc[:, 'yield'] = np.average(rice_yield[models], axis=1)
    results=round(results,2)
    
    a=pd.merge(lonlat,results,how='left',on=['x','y'])
    
    CH4=np.array(a['CH4']).reshape(2160,4320)
    N2O=np.array(a['N2O']).reshape(2160,4320)
    yields=np.array(a['yield']).reshape(2160,4320)
    
    data=[CH4,N2O,yields]
    return data


# In[277]:


for i in range(0,8):
    start=time.time()
    data=future2npy(i)
    filename = os.path.join("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/"+files[i]+'.npy')
    np.save(filename, data)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))


# In[278]:


np.nanmean(np.load("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/"+files[7]+'.npy')[0])


# # 绘制四个时间 SSP126和SSP585的second season prediction

# In[128]:


def future_scenario_prediction(paths,i):
    temp_rice=pd.read_csv(os.path.join(paths[i]+'/temp_rice.csv'))
    prec_rice=pd.read_csv(os.path.join(paths[i]+'/prec_rice.csv'))
    
    print(str(i)+" 气候降水数据读取")
    #修改列名
    newcolumns=['longitude', 'latitude']
    for c in range(2,len(temp_rice.columns)):
        #print(temp_rice.columns[i])
        n=temp_rice.columns[c][14:-21]  ##选择出模式名
        newcolumns.append(n)
    temp_rice.columns=newcolumns
    prec_rice.columns=newcolumns
    
    #补充缺失值
    #上下两个值的平均值进行填充
    for data in [temp_rice,prec_rice]:
        data.fillna(data.interpolate(),inplace=True)
        
    print("缺失值填充")
    start = time.time()
    rice=joblib.load("rice.pkl")
    
    rice=rice.iloc[:,0:30]
    
    rice_CH4=[]
    rice_N2O=[]
    rice_yield=[]
    
    for model in models:
        rice["temp"]=temp_rice[model]
        rice["prec"]=prec_rice[model]
        #name=names[k]
        rice_CH4.append(rice_cal(rice,xgbr_CH4,scaler_CH4,attributes2))
        rice_N2O.append(rice_cal(rice,xgbr_N2O,scaler_N2O,attributes2))
        rice_yield.append(rice_cal(rice,xgbr_yield,scaler_yield,attributes2))
    end = time.time()
    print(' Running time: %s Seconds'%(end-start))
    
    file_names=["rice_CH4","rice_N2O","rice_yield"]
    out_data=[rice_CH4,rice_N2O,rice_yield]
    #names=temp_rice.columns[2:temp_single_rice.shape[1]+2]
    start =time.time()
    for j in range(3):
        data=out_data[j]
        data=pd.DataFrame(data).T
        data.columns=models
        data['x']=rice['x']
        data['y']=rice['y']
        data['area']=rice['area']
        data.to_csv(os.path.join(paths[i]+'/pred2/'+file_names[j]+".csv"))
    end = time.time()
    print('Save Running time: %s Seconds'%(end-start))


# In[129]:


for i in range(0,8):
    future_scenario_prediction(paths,i)


# In[165]:


paths


# In[268]:


lonlat


# In[162]:


###转npy文件
def future2npy(i):
    rice_CH4=pd.read_csv(os.path.join(paths[i]+'/pred2/rice_CH4.csv'))
    rice_N2O=pd.read_csv(os.path.join(paths[i]+'/pred2/rice_N2O.csv'))
    rice_yield=pd.read_csv(os.path.join(paths[i]+'/pred2/rice_yield.csv'))
    results=rice_CH4[['x','y']]
    
    results.loc[:, 'CH4'] = np.average(rice_CH4[models], axis=1)
    results.loc[:, 'N2O'] = np.average(rice_N2O[models], axis=1)
    results.loc[:, 'yield'] = np.average(rice_yield[models], axis=1)
    results=round(results,2)
    
    a=pd.merge(lonlat,results,how='left',on=['x','y'])
    
    
    CH4=np.array(a['CH4']).reshape(2160,4320)
    N2O=np.array(a['N2O']).reshape(2160,4320)
    yields=np.array(a['yield']).reshape(2160,4320)
    
    data=[CH4,N2O,yields]
    return data


# In[169]:


for i in range(0,8):
    start=time.time()
    data=future2npy(i)
    filename = os.path.join("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/"+files[i]+'_2.npy')
    np.save(filename, data)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))


# In[264]:


files


# In[279]:


np.nanmean(np.load("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/"+'2081-2100ssp126'+".npy")[0])


# # 绘图

# In[75]:


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

from osgeo import gdal
from osgeo import gdal, osr
import matplotlib


# In[26]:


config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 10, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['Arial'], # 'Simsun'宋体
    "axes.unicode_minus": False,# 用来正常显示负号
}
plt.rcParams.update(config)


# In[367]:


filepath1="D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/"+'2081-2100ssp126'+'.npy'
filepath2="D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/"+'2081-2100ssp585'+'.npy'

filepath="D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/global_prediction.npy"
##baseline
data1=np.load(filepath)[0]
data2=np.load(filepath)[1]
data3=np.load(filepath)[2]

##SSP126
data4=np.load(filepath1)[0]
data5=np.load(filepath1)[1]
data6=np.load(filepath1)[2]
##SSP585
data7=np.load(filepath2)[0]
data8=np.load(filepath2)[1]
data9=np.load(filepath2)[2]

s1=(data4-data1)/data1*100
s2=(data5-data2)/data2*100
s3=(data6-data3)/data3*100
s4=(data7-data1)/data1*100
s5=(data8-data2)/data2*100
s6=(data9-data3)/data3*100

s1=np.clip(s1,-100,100)
s2=np.clip(s2,-100,100)
s3=np.clip(s3,-100,100)
s4=np.clip(s4,-100,100)
s5=np.clip(s5,-100,100)
s6=np.clip(s6,-100,100)


# In[291]:


np.nanpercentile(s3,1)


# In[319]:


for i in range(0,100,10):
    print(i)
    print(np.nanpercentile(s3,i))


# In[368]:


fig=plt.figure(figsize=(8,8))
##rect可以设置子图的位置与大小
rect1 = [0, 0.95, 0.45, 0.3]
# [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例），左下角为(0,0),右上角为(1,1)
rect2 = [0, 0.65, 0.45, 0.3]
rect3 = [0, 0.35, 0.45, 0.3]
rect4 = [0.5, 0.95, 0.45, 0.3]
rect5 = [0.5, 0.65, 0.45, 0.3]
rect6 = [0.5, 0.35, 0.45, 0.3]

x = np.linspace(-180,180, data1.shape[1])
y = np.linspace(90,-90, data1.shape[0])

xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵

bounds=[-100, -80, -60,-40,-20,0, 20,40, 60, 80, 100]
cmap = plt.get_cmap('coolwarm')
ticks = bounds
norm = colors.BoundaryNorm(bounds, cmap.N)
#在fig中添加子图ax，并赋值位置rect
ax1 = plt.axes(rect1)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawlsmask(land_color='1')#绘制陆地海洋掩膜

m.contourf(xx, yy, s1,cmap=cmap,norm=norm,latlon=True)
cb1 = m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb1.set_label('Change in C$\mathregular{H_4}$ (%)', fontsize=12)

bounds=[-100, -80, -60,-40,-20,0, 20,40, 60, 80, 100]
cmap = plt.get_cmap('coolwarm')
ticks = bounds
norm = colors.BoundaryNorm(bounds, cmap.N)

ax2 = plt.axes(rect2)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawlsmask(land_color='1')#绘制陆地海洋掩膜
m.contourf(xx, yy, s2,cmap=cmap,norm=norm,latlon=True)
cb2 = m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb2.set_label('Change in $\mathregular{N_2}$O (%)', fontsize=12)


bounds=[-40,-32,-24,-16,-8,0,8,16,24,32,40]
cmap = plt.get_cmap('coolwarm')
ticks = bounds
norm = colors.BoundaryNorm(bounds, cmap.N)

ax3 = plt.axes(rect3)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawlsmask(land_color='1')#绘制陆地海洋掩膜

m.contourf(xx, yy, s3,cmap=cmap,norm=norm,latlon=True)
m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb3 = m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb3.set_label('Change in Yield (%)', fontsize=12)


bounds=[-100, -80, -60,-40,-20,0, 20,40, 60, 80, 100]
cmap = plt.get_cmap('coolwarm')
ticks = bounds
norm = colors.BoundaryNorm(bounds, cmap.N)


ax4 = plt.axes(rect4)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawlsmask(land_color='1')#绘制陆地海洋掩膜
m.contourf(xx, yy, s4,cmap=cmap,norm=norm,latlon=True)
m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb4 = m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb4.set_label('Change in C$\mathregular{H_4}$ (%)', fontsize=12)


bounds=[-100, -80, -60,-40,-20,0, 20,40, 60, 80, 100]
cmap = plt.get_cmap('coolwarm')
ticks = bounds
norm = colors.BoundaryNorm(bounds, cmap.N)

ax5 = plt.axes(rect5)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawlsmask(land_color='1')#绘制陆地海洋掩膜
m.contourf(xx, yy, s5,cmap=cmap,norm=norm,latlon=True)
m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb5 = m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb5.set_label('Change in $\mathregular{N_2}$O (%)', fontsize=12)


bounds=[-40,-32,-24,-16,-8,0,8,16,24,32,40]
cmap = plt.get_cmap('coolwarm')
ticks = bounds
norm = colors.BoundaryNorm(bounds, cmap.N)
ax6 = plt.axes(rect6)
m = Basemap(llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=90)
m.drawcoastlines(linewidth=0.5)  # 画海岸线
m.drawcountries()
m.drawlsmask(land_color='1')#绘制陆地海洋掩膜
m.contourf(xx, yy, s6,cmap=cmap,norm=norm,latlon=True)
m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom',)
cb6 = m.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
           extend='both',size='4%',pad='4%',ticks=ticks,location='bottom')
cb6.set_label('Change in Yield (%)', fontsize=12)

ax1.set_title("a",loc='left',fontweight='bold',fontsize=15)
ax2.set_title("b",loc='left',fontweight='bold',fontsize=15)
ax3.set_title("c",loc='left',fontweight='bold',fontsize=15)
ax4.set_title("d",loc='left',fontweight='bold',fontsize=15)
ax5.set_title("e",loc='left',fontweight='bold',fontsize=15)
ax6.set_title("f",loc='left',fontweight='bold',fontsize=15)


# In[370]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\climate_change_effects.png",dpi=600,bbox_inches='tight')


# # boxplot＋climate change per country

# In[141]:


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


# In[ ]:


# 箱线图，使用收获面积*主要季节数据近似计算


# In[379]:


files=os.listdir('D:/li/temporary/future_climate_extract/')
#print(type(files))
#print(files)
paths=[]
for i in files:
    path=os.path.join('D:/li/temporary/future_climate_extract/'+i)
    paths.append(path)
print(paths)


# In[380]:


models=['CNRM-CM6-1','CanESM5','CNRM-CM6-1-HR','EC-Earth3-Veg','GISS-E2-1-G','GISS-E2-1-H',
        'HadGEM3-GC31-LL','INM-CM4-8','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-HR','UKESM1-0-LL']


# In[381]:


results=[]
def tot_cal(i):
    rice_CH4 = pd.read_csv(os.path.join(paths[i]+'/pred/rice_CH4.csv'))
    rice_N2O = pd.read_csv(os.path.join(paths[i]+'/pred/rice_N2O.csv'))
    rice_yield = pd.read_csv(os.path.join(paths[i]+'/pred/rice_yield.csv'))

    for m in models:
        tot_CH4=sum(rice_CH4[m]*rice_CH4['area'])
        tot_N2O=sum(rice_N2O[m]*rice_N2O['area'])
        tot_yield=sum(rice_yield[m]*rice_yield['area'])
        results.append([files[i][0:9],files[i][9:15],tot_CH4,tot_N2O,tot_yield])


# In[382]:


for i in range(0,8):
    tot_cal(i)
results=pd.DataFrame(results)
results.columns=['time','SSP','CH4','N2O','yield']


# In[383]:


results['CH4_%']=100*(results['CH4']-sum(rice['CH4_xgb']))/sum(rice['CH4_xgb'])
results['N2O_%']=100*(results['N2O']-sum(rice['N2O_xgb']))/sum(rice['N2O_xgb'])
results['yield_%']=100*(results['yield']-sum(rice['yield_xgb']))/sum(rice['yield_xgb'])


# In[385]:


results[(results['SSP']=='ssp126')&(results['time']=='2081-2100')].describe()


# In[386]:


results[(results['SSP']=='ssp585')&(results['time']=='2081-2100')].describe()


# In[ ]:





# In[330]:


results=results.replace('ssp126','SSP126')
results=results.replace('ssp585','SSP585')


# In[331]:


results


# In[333]:


results.to_csv('boxplot_data.csv')


# In[332]:


fig=plt.figure(figsize=(12,4))
rect1 = [0.10, 0.95, 0.2, 0.8] 
# [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例），左下角为(0,0),右上角为(1,1)
rect2 = [0.40, 0.95, 0.2, 0.8]
rect3 = [0.70, 0.95, 0.2, 0.8]
#plt.boxplot(x=CH4)
ax1=plt.axes(rect1)
sns.boxplot(x='time', y='CH4_%', hue='SSP', data=results ,palette=['#7fbf7b','#af8dc3'], 
            linewidth = 1 , fliersize = 1,ax=ax1,saturation=1)
plt.xlabel("")
plt.xticks(rotation=30)
plt.ylabel("Projected change of C$\mathregular{H_4}$ (%)")
ax1.legend(loc=[0.10,0.70])

ax2=plt.axes(rect2)
sns.boxplot(x='time', y='N2O_%',hue='SSP', data=results , palette=['#7fbf7b','#af8dc3'], 
            linewidth = 1 , fliersize = 1,ax=ax2,saturation=1)
plt.xlabel("")
plt.xticks(rotation=30)
plt.ylabel("Projected change of $\mathregular{N_2}$O (%)")
ax2.legend_.remove()

ax3=plt.axes(rect3)
sns.boxplot(x='time', y='yield_%',hue='SSP', data=results, palette=['#7fbf7b','#af8dc3'], 
            linewidth =1 , fliersize = 1,ax=ax3, saturation=1)
plt.xlabel("")
plt.xticks(rotation=30)
plt.ylabel("Projected change of yield (%)")
ax3.legend_.remove()

ax1.set_title("a",loc='left',fontweight='bold')
ax2.set_title("b",loc='left',fontweight='bold')
ax3.set_title("c",loc='left',fontweight='bold')
plt.show()


# In[157]:


# 将rice中duration2列保存为tif
rice.rename(columns={'longitude': 'x', 'latitude': 'y'}, inplace=True)
a = pd.merge(lonlat,rice,how='left',on=['x','y'])
duration2 = np.array(a['duration2']).reshape(2160,4320)


from osgeo import gdal, osr
import numpy as np

# 创建 GeoTIFF 文件
def create_geotiff(filename, data, reference_tiff):
    # 加载参考文件
    reference_dataset = gdal.Open(reference_tiff)

    # 获取参考文件的投影信息和地理转换参数
    projection = reference_dataset.GetProjection()
    geotransform = reference_dataset.GetGeoTransform()

    # 获取参考文件的行数、列数和波段数
    rows, cols = data.shape
    bands = 1

    # 创建新的 GeoTIFF 文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, cols, rows, bands, gdal.GDT_Float32)

    # 设置地理转换参数
    dataset.SetGeoTransform(geotransform)

    # 设置投影信息
    dataset.SetProjection(projection)

    # 写入数据
    dataset.GetRasterBand(1).WriteArray(data)

    # 关闭数据集
    dataset = None

# 调用函数创建 GeoTIFF 文件
create_geotiff('D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/duration2.tif', 
               duration2, 
               'D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif')


# In[110]:


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


# In[111]:


files=os.listdir('D:/li/mapdata/countrydata')
filenames=[]
for i in range(len(files)):
    path=os.path.join('D:/li/mapdata/countrydata/'+files[i]+'/')
    #paths.append(path)
    name=os.path.join(files[i][0:-4]+'_0.shp')
    filename=os.path.join(path+name)
    filenames.append(filename)


# In[170]:


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


# In[293]:


np.nanmean(ssp126_2100[0])


# In[294]:


filepath1 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2081-2100ssp126_2.npy"
filepath2 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2081-2100ssp585_2.npy"
ssp126_2100 = np.load(filepath1)
ssp585_2100 = np.load(filepath2)

if __name__ == '__main__':
    arrs = [ssp126_2100,ssp585_2100]
    names=["ssp126_2100_2.tif","ssp585_2100_2.tif"]
         
    src_ras_file = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"  
    # 提供地理坐标信息和几何信息的栅格底图
    dataset = gdal.Open(src_ras_file)
    projection = dataset.GetProjection()
    transform = dataset.GetGeoTransform()
    for i in range(2):
        arr=arrs[i]
        raster_file = os.path.join("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/",names[i])
         # 输出的栅格文件路径
        arr2raster(arr, raster_file, prj=projection, trans=transform)


# In[295]:


filepath1 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2081-2100ssp126.npy"
filepath2 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/2081-2100ssp585.npy"
ssp126_2100 = np.load(filepath1)
ssp585_2100 = np.load(filepath2)

if __name__ == '__main__':
    arrs = [ssp126_2100,ssp585_2100]
    names=["ssp126_2100.tif","ssp585_2100.tif"]
         
    src_ras_file = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"  
    # 提供地理坐标信息和几何信息的栅格底图
    dataset = gdal.Open(src_ras_file)
    projection = dataset.GetProjection()
    transform = dataset.GetGeoTransform()
    for i in range(2):
        arr=arrs[i]
        raster_file = os.path.join("D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/",names[i])
         # 输出的栅格文件路径
        arr2raster(arr, raster_file, prj=projection, trans=transform)


# In[296]:


rice.describe()


# In[304]:


def country_sum(i):
    shpPath = filenames[i]

# #2081-2100 SSP585
#     imgPath1 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/ssp585_2100.tif"
#     imgPath2 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/ssp585_2100_2.tif"
    
#2081-2100 SSP126
    imgPath1 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/ssp126_2100.tif"
    imgPath2 = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/ssp126_2100_2.tif"
    
    harvested_areapath = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"
    physical_areapath = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_physical-area.geotiff/spam2000v3r7_physical-area_RICE.tif"
    duration2_path = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/duration2.tif"
    
    data1 = clip(shpPath, imgPath1)
    data2 = clip(shpPath, imgPath2)
    harvested_area = clip(shpPath,harvested_areapath)
    physical_area = clip(shpPath,physical_areapath)
    duration2 = clip(shpPath,duration2_path)
    

    # 计算 CH4_xgb 的值
    CH4_xgb = np.where(duration2 > 0, 
                       data1[0] * physical_area + data2[0] * (harvested_area - physical_area),
                       data1[0] * harvested_area)

    # 计算 N2O_xgb 的值
    N2O_xgb = np.where(duration2 > 0,
                       data1[1] * physical_area + data2[1] * (harvested_area - physical_area),
                       data1[1] * harvested_area)

    # 计算 yields_xgb 的值
    yields_xgb = np.where(duration2 > 0,
                          data1[2] * physical_area + data2[2] * (harvested_area - physical_area),
                          data1[2] * harvested_area)
    

    # 计算总量
    CH4 = np.nansum(CH4_xgb)
    N2O = np.nansum(N2O_xgb)
    yields = np.nansum(yields_xgb)
    shps = shapefile.Reader(shpPath)
    abb=shps.shapeRecord().record[0]
    country=shps.shapeRecord().record[1]
    area=np.nansum(harvested_area)
    result=[abb,country,area,CH4,N2O,yields]
    return result


# In[302]:


results=[]
for i in range(len(filenames)):
    start =time.time()
    s=country_sum(i)
    results.append(s)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))

future585=pd.DataFrame(results)
future585.columns=['abb','country','area','CH4','N2O','yields']


# In[303]:


future585


# In[305]:


results=[]
for i in range(len(filenames)):
    start =time.time()
    s=country_sum(i)
    results.append(s)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))

future126=pd.DataFrame(results)
future126.columns=['abb','country','area','CH4','N2O','yields']


# In[306]:


future126


# In[307]:


def country_sum_baseline(i):
    shpPath = filenames[i]
    
    imgPath = "D:/li/codesdata/rice/python_calculate/NEW_V2/future_prediction_npy/global_prediction_total.tif"
    harvested_areapath = "D:/li/codesdata/rice/data/SPAM/spam2000v3.0.7_global_harvested-area.geotiff/spam2000v3r7_harvested-area_RICE.tif"
    
    data = clip(shpPath, imgPath)
    harvested_area = clip(shpPath,harvested_areapath)
    
    CH4=np.nansum(data[0])
    N2O=np.nansum(data[1])
    yields=np.nansum(data[2])
    
    area=np.nansum(harvested_area)
    
    shps = shapefile.Reader(shpPath)
    abb=shps.shapeRecord().record[0]
    country=shps.shapeRecord().record[1]
    result=[abb,country,area,CH4,N2O,yields]
    return result


# In[308]:


results=[]
for i in range(len(filenames)):
    start =time.time()
    s=country_sum_baseline(i)
    results.append(s)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))

pred_2000=pd.DataFrame(results)
pred_2000.columns=['abb','country','area','CH4','N2O','yields']


# In[309]:


pred_2000


# In[311]:


writer = pd.ExcelWriter('future_per_country.xlsx', engine='openpyxl') # 创建ExcelWriter对象
future126.to_excel(writer, sheet_name='future126') # 按照writer设定的参数方式写入第一个数据框
future585.to_excel(writer, sheet_name='future585') # 按照writer设定的参数方式写入第二个数据框
pred_2000.to_excel(writer, sheet_name='pred2000')
writer.save()


# In[312]:


# 变化值，
df126=pred_2000[['abb','country','area']]
df126['CH4_change']=future126['CH4']-pred_2000['CH4']
df126['N2O_change']=future126['N2O']-pred_2000['N2O']
df126['yield_change']=future126['yields']-pred_2000['yields']

df126['CH4_change_ratio']=(future126['CH4']-pred_2000['CH4'])/pred_2000['CH4']
df126['N2O_change_ratio']=(future126['N2O']-pred_2000['N2O'])/pred_2000['N2O']
df126['yield_change_ratio']=(future126['yields']-pred_2000['yields'])/pred_2000['yields']


# In[313]:


df585=pred_2000[['abb','country','area']]
df585['CH4_change']=future585['CH4']-pred_2000['CH4']
df585['N2O_change']=future585['N2O']-pred_2000['N2O']
df585['yield_change']=future585['yields']-pred_2000['yields']

df585['CH4_change_ratio']=(future585['CH4']-pred_2000['CH4'])/pred_2000['CH4']
df585['N2O_change_ratio']=(future585['N2O']-pred_2000['N2O'])/pred_2000['N2O']
df585['yield_change_ratio']=(future585['yields']-pred_2000['yields'])/pred_2000['yields']


# In[345]:


df126


# In[349]:


countries = df126['abb']


# In[317]:


import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
countries = df126['abb']


# 设置图像大小
fig=plt.figure(figsize=(16, 4))

# 绘制第一个子图
data1 = df126['CH4_change']/1000000
data2 = df585['CH4_change']/1000000
plt.subplot(1, 3, 1)
plt.bar(np.arange(16) - 0.2, data1, width=0.4, color='#7fbf7b', label='SSP126')
plt.bar(np.arange(16) + 0.2, data2, width=0.4, color='#af8dc3', label='SSP585')
plt.xticks(np.arange(16), countries, rotation=90)
plt.ylabel("C$\mathregular{H_4}$ changes (Mt year$^{-1}$)")
plt.title('a',loc='left',fontweight='bold',fontsize=20)
plt.legend()

# 绘制第二个子图
data1 = df126['N2O_change']/1000000
data2 = df585['N2O_change']/1000000
plt.subplot(1, 3, 2)
plt.bar(np.arange(16) - 0.2, data1, width=0.4, color='#1b7837', label='SSP126')
plt.bar(np.arange(16) + 0.2, data2, width=0.4, color='#762a83', label='SSP585')
plt.xticks(np.arange(16), countries, rotation=90)
plt.ylabel("$\mathregular{N_2}$O changes (Mt N year$^{-1}$)")
plt.title('b',loc='left',fontweight='bold',fontsize=20)
# plt.legend()

# 绘制第三个子图
data1 = df126['yield_change']/1000000
data2 = df585['yield_change']/1000000
plt.subplot(1, 3, 3)
plt.bar(np.arange(16) - 0.2, data1, width=0.4, color='#f1a340', label='SSP126')
plt.bar(np.arange(16) + 0.2, data2, width=0.4, color='#998ec3', label='SSP585')
plt.xticks(np.arange(16), countries, rotation=90)
plt.ylabel("Yield changes (Mt year$^{-1}$)")
plt.title('c',loc='left',fontweight='bold',fontsize=20)
# plt.legend()

plt.show()


# # 箱线图+柱形图

# In[371]:


results = pd.read_csv('boxplot_data.csv')


# In[372]:


df126 = df126.sort_values(by='area', ascending=False)
df585 = df585.sort_values(by='area', ascending=False)


# In[377]:


results['time']


# In[373]:


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


# In[374]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置图像大小
fig, axes = plt.subplots(2, 3, figsize=(11, 7))

# 绘制第一行第一个子图
sns.boxplot(x='time', y='CH4_%', hue='SSP', data=results, palette=['#7fbf7b', '#af8dc3'],
            linewidth=1, fliersize=1, ax=axes[0, 0], saturation=1)
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Projected change of C$\mathregular{H_4}$ (%)")
axes[0, 0].set_title('a', loc='left', fontweight='bold')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=30) 


# 绘制第一行第二个子图
sns.boxplot(x='time', y='N2O_%', hue='SSP', data=results, palette=['#7fbf7b', '#af8dc3'],
            linewidth=1, fliersize=1, ax=axes[0, 1], saturation=1)
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Projected change of $\mathregular{N_2}$O (%)")
axes[0, 1].set_title('b', loc='left', fontweight='bold')
axes[0, 1].legend_.remove()
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=30)  # 设置标签倾斜


# 绘制第一行第三个子图
sns.boxplot(x='time', y='yield_%', hue='SSP', data=results, palette=['#7fbf7b', '#af8dc3'],
            linewidth=1, fliersize=1, ax=axes[0, 2], saturation=1)
axes[0, 2].set_xlabel("")
axes[0, 2].set_ylabel("Projected change of yield (%)")
axes[0, 2].set_title('c', loc='left', fontweight='bold')
axes[0, 2].legend_.remove()
axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=30)  # 设置标签倾斜


# 绘制第二行第一个子图
data1 = df126['CH4_change'] / 1000000
data2 = df585['CH4_change'] / 1000000
axes[1, 0].bar(np.arange(16) - 0.2, data1, width=0.4, color='#7fbf7b', label='SSP126')
axes[1, 0].bar(np.arange(16) + 0.2, data2, width=0.4, color='#af8dc3', label='SSP585')
axes[1, 0].set_xticks(np.arange(16))
axes[1, 0].set_xticklabels(countries, rotation=90)
axes[1, 0].set_ylabel("C$\mathregular{H_4}$ changes (Mt year$^{-1}$)")
axes[1, 0].set_title('d', loc='left', fontweight='bold')

# 绘制第二行第二个子图
data1 = df126['N2O_change'] / 1000000
data2 = df585['N2O_change'] / 1000000
axes[1, 1].bar(np.arange(16) - 0.2, data1, width=0.4, color='#7fbf7b', label='SSP126')
axes[1, 1].bar(np.arange(16) + 0.2, data2, width=0.4, color='#af8dc3', label='SSP585')
axes[1, 1].set_xticks(np.arange(16))
axes[1, 1].set_xticklabels(countries, rotation=90)
axes[1, 1].set_ylabel("$\mathregular{N_2}$O changes (Mt N year$^{-1}$)")
axes[1, 1].set_title('e', loc='left', fontweight='bold')
# axes[1, 1].legend_.remove()

# 绘制第二行第三个子图
data1 = df126['yield_change'] / 1000000
data2 = df585['yield_change'] / 1000000
axes[1, 2].bar(np.arange(16) - 0.2, data1, width=0.4, color='#7fbf7b', label='SSP126')
axes[1, 2].bar(np.arange(16) + 0.2, data2, width=0.4, color='#af8dc3', label='SSP585')
axes[1, 2].set_xticks(np.arange(16))
axes[1, 2].set_xticklabels(countries, rotation=90)
axes[1, 2].set_ylabel("Yield changes (Mt year$^{-1}$)")
axes[1, 2].set_title('f', loc='left', fontweight='bold')
# axes[1, 2].legend_.remove()

plt.tight_layout()
plt.show()


# In[387]:


df126


# In[388]:


df585


# In[375]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\future_climate_changes.png",dpi=600,bbox_inches='tight')

