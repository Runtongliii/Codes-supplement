#!/usr/bin/env python
# coding: utf-8

# In[3]:


from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import copy
#python中赋值，如果是可变对象，对其中一个修改会影响到另一个。如果要生成完全新的对象，应使用deepcopy
import joblib
import time
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tqdm import tqdm


# In[4]:


rice = joblib.load('rice.pkl')
rice['water_regime_AWD']=rice['AWD']*rice['irrigated_fraction']
rice['water_regime_continuously flooding']=rice['CF']*rice['irrigated_fraction']
rice['water_regime_midseason drainage']=rice['MSD']*rice['irrigated_fraction']
rice['water_regime_rainfed']=rice['rainfed_fraction']


# In[36]:


rice


# In[3]:


attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']
num_attr = ['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin']


# In[4]:


xgbr_CH4 = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=400)
xgbr_N2O = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=200)
xgbr_yield = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=500)


# In[5]:


CH4 = pd.read_csv('CH4.csv', index_col=0) 
N2O = pd.read_csv('N2O.csv', index_col=0) 
yields = pd.read_csv('yields.csv', index_col=0) 


# In[6]:


# 定义蒙特卡罗模拟次数
num_simulations = 1000

# # 在每个特征上添加随机噪声，这样不同数值型
# def add_noise_to_data(data, noise_level=0.1):
#     noisy_data = data.copy()
#     for col in noisy_data.columns:
#         if col in num_attr:  # 只对数值型特征添加噪声
#             noise = np.random.normal(loc=0, scale=noise_level, size=len(noisy_data))
#             noisy_data[col] += noise
#     return noisy_data


#在每个特征上添加随机噪声，修改比例
def add_noise_to_data(data, noise_level=0.1):
    noisy_data = data.copy()
    for col in noisy_data.columns:
        if col in num_attr:  # 只对数值型特征添加噪声
            noise = np.random.normal(loc=0, scale=1, size=len(noisy_data))
            noisy_data[col] *= (1 + noise_level*noise)
    return noisy_data


# 生成随机参数
def generate_random_params(param_ranges):
    params = {}

    for param, (lower, upper) in param_ranges.items():
        params[param] = np.random.uniform(lower, upper)
    # max_depth 和 n_estimators 参数需要是整数
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    return params

# # 训练模型并预测产量/排放，
# def train_and_predict_label(x_train, y_train, params,predict_data):
#     model = xgb.XGBRegressor(**params)
#     scaler = StandardScaler()
#     x_train[num_attr] = scaler.fit_transform(x_train[num_attr])
#     model.fit(x_train, y_train)
    
    
#     predict_data_copy = predict_data.copy()
#     predict_data_copy.loc[:, num_attr] = scaler.transform(predict_data_copy[num_attr])
#     predicted_label = model.predict(predict_data_copy)  # 假设 predict_data 是您的预测数据
    
#     return predicted_label

# 训练模型并预测产量/排放，同时添加不确定性
def train_and_predict_label_with_noise(x_train, y_train, params, predict_data, noise_level=0.1):
    model = xgb.XGBRegressor(**params)
    scaler = StandardScaler()
    x_train[num_attr] = scaler.fit_transform(x_train[num_attr])
    model.fit(x_train, y_train)
    
    # 创建带有不确定性的预测数据副本
    predict_data_copy = add_noise_to_data(predict_data, noise_level)
    predict_data_copy.loc[:, num_attr] = scaler.transform(predict_data_copy[num_attr])
    
    # 进行预测
    predicted_label = model.predict(predict_data_copy)
    
    return predicted_label


# 生成训练数据的不确定性
def generate_random_train_data(data,label, i):
    x_train, x_test, y_train, y_test = train_test_split(data[attributes],data[label], test_size=0.2,random_state=i)
    return x_train, y_train


# In[7]:


# 蒙特卡罗模拟甲烷的不确定性
best_params = {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 400}
#生成模型超参数的不确定性
param_ranges = {
    'learning_rate': (best_params['learning_rate'] - 0.05, best_params['learning_rate'] + 0.05),
    'max_depth': (best_params['max_depth'] - 2, best_params['max_depth'] + 2),
    'n_estimators': (best_params['n_estimators'] - 100, best_params['n_estimators'] + 100)
}

CH4_simulations = []
data = CH4
label = 'CH4'
predict_data = rice[attributes]

for _ in tqdm(range(num_simulations), desc="Simulating CH4", unit="iteration"):
    params = generate_random_params(param_ranges)
    x_train, y_train = generate_random_train_data(data,label, _)
    predicted_CH4 = train_and_predict_label_with_noise(x_train, y_train, params,predict_data)  # 假设 rice_attributes 是您的预测数据
    CH4_simulations.append(predicted_CH4)


# In[18]:


(np.std(CH4_simulations, axis=0)/np.mean(CH4_simulations, axis=0)).mean()


# In[9]:


# 蒙特卡罗模拟氧化亚氮的不确定性
best_params = {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}

#生成模型超参数的不确定性
param_ranges = {
    'learning_rate': (best_params['learning_rate'] - 0.05, best_params['learning_rate'] + 0.05),
    'max_depth': (best_params['max_depth'] - 2, best_params['max_depth'] + 2),
    'n_estimators': (best_params['n_estimators'] - 100, best_params['n_estimators'] + 100)
}
N2O_simulations = []
data = N2O
label = 'N2O'
predict_data = rice[attributes]

for _ in tqdm(range(num_simulations), desc="Simulating N2O", unit="iteration"):
    params = generate_random_params(param_ranges)
    x_train, y_train = generate_random_train_data(data,label, _)
    predicted_N2O = train_and_predict_label_with_noise(x_train, y_train, params,predict_data)  # 假设 rice_attributes 是您的预测数据
    N2O_simulations.append(predicted_N2O)


# In[10]:


(np.std(N2O_simulations, axis=0)/np.mean(N2O_simulations, axis=0)).mean()


# In[11]:


# 蒙特卡罗模拟产量的不确定性
best_params = {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 500}

#生成模型超参数的不确定性
param_ranges = {
    'learning_rate': (best_params['learning_rate'] - 0.05, best_params['learning_rate'] + 0.05),
    'max_depth': (best_params['max_depth'] - 2, best_params['max_depth'] + 2),
    'n_estimators': (best_params['n_estimators'] - 100, best_params['n_estimators'] + 100)
}

yield_simulations = []
data = yields
label = 'yield'
predict_data = rice[attributes]

for _ in tqdm(range(num_simulations), desc="Simulating yield", unit="iteration"):
    params = generate_random_params(param_ranges)
    x_train, y_train = generate_random_train_data(data,label, _)
    predicted_yield = train_and_predict_label_with_noise(x_train, y_train, params,predict_data)  # 假设 rice_attributes 是您的预测数据
    yield_simulations.append(predicted_yield)


# In[16]:


(np.std(yield_simulations, axis=0)/np.mean(yield_simulations, axis=0)).mean()


# In[21]:


import numpy as np
import pandas as pd
# 定义一个函数来计算统计信息
def calculate_statistics(simulations):
    means = np.mean(simulations, axis=0)
    std_devs = np.std(simulations, axis=0)
    uncertainties = std_devs / means
    confidence_interval_lower = np.percentile(simulations, 2.5, axis=0)
    confidence_interval_upper = np.percentile(simulations, 97.5, axis=0)
    return pd.DataFrame({
        'Mean': means,
        'Uncertainty': uncertainties,
        'Lower_CI': confidence_interval_lower,
        'Upper_CI': confidence_interval_upper
    })

# 分别计算每个变量的统计信息并保存到数据框中
yield_df = calculate_statistics(yield_simulations)
N2O_df = calculate_statistics(N2O_simulations)
CH4_df = calculate_statistics(CH4_simulations)

# 保存结果到 Excel 文件中
with pd.ExcelWriter('simulation_results.xlsx') as writer:
    yield_df.to_excel(writer, sheet_name='Yield', index=False)
    N2O_df.to_excel(writer, sheet_name='N2O', index=False)
    CH4_df.to_excel(writer, sheet_name='CH4', index=False)

print("结果已保存到 simulation_results.xlsx 文件中。")


# # 绘图

# In[5]:


import pandas as pd

# 读取 Excel 文件
with pd.ExcelFile('simulation_results.xlsx') as xls:
    # 读取 'Yield' 工作表中的数据框
    yield_df = pd.read_excel(xls, sheet_name='Yield')
    
    # 读取 'N2O' 工作表中的数据框
    N2O_df = pd.read_excel(xls, sheet_name='N2O')
    
    # 读取 'CH4' 工作表中的数据框
    CH4_df = pd.read_excel(xls, sheet_name='CH4')


# In[8]:


rice = joblib.load('rice.pkl')


# In[9]:


# 创建一个空的 DataFrame，用于存储合并后的结果
merged_df = pd.DataFrame()

# 合并 "rice" 数据集的 "longitude" 和 "latitude" 列
merged_df['longitude'] = rice['longitude']
merged_df['latitude'] = rice['latitude']

# 合并 "Yield"、"N2O" 和 "CH4" 数据集的列
prefixes = ['yield_', 'N2O_', 'CH4_']
dfs = [yield_df, N2O_df, CH4_df]

for prefix, df in zip(prefixes, dfs):
    for column in df.columns:
        merged_df[prefix + column] = df[column]


# In[10]:


merged_df.describe()


# In[ ]:


# 计算总量


# In[ ]:


# 保存成为npy文件,计算总量和95%置信区间使用这个数据


# In[11]:


# 导入经纬度
lonlat = pd.read_csv(r"D:\li\codesdata\rice\data\lonlat.csv")
lonlat.rename(columns={lonlat.columns[0]:'longitude'},inplace=True)
lonlat.rename(columns={lonlat.columns[1]:'latitude'},inplace=True)
lonlat=round(lonlat,2)


# In[43]:


merged_df.describe()


# In[43]:


start=time.time()
a=pd.merge(lonlat,merged_df,how='left',on=['longitude','latitude'])
CH4_Uncertainty = np.array(a['CH4_Uncertainty']).reshape(2160,4320)
N2O_Uncertainty = np.array(a['N2O_Uncertainty']).reshape(2160,4320)
yield_Uncertainty = np.array(a['yield_Uncertainty']).reshape(2160,4320)

Uncertainties = [CH4_Uncertainty,N2O_Uncertainty,yield_Uncertainty]

end = time.time()
print('Running time: %s Seconds'%(end-start))


# # Map

# In[14]:


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


# In[22]:


from osgeo import gdal
from osgeo import gdal, osr
import matplotlib


# In[20]:


import matplotlib.pyplot as plt

# Before calling pcolor, set the rcParams for pcolor.shading
plt.rcParams['pcolor.shading'] = 'auto'

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


# In[44]:


Uncertainties = [arr * 100 for arr in Uncertainties]


# In[46]:


fig,axs = plt.subplots(3,1)
fig.set_size_inches(6,9)
x = np.linspace(-180,180, 4320)
y = np.linspace(90,-90, 2160)

labs = ['a','b','c']

bounds=[0,10,20,30,40,50,60,70,80,90,100]
cmap = plt.get_cmap('Spectral_r',10)

min = 0
max = 100
norm1 = colors.BoundaryNorm(bounds,10)
xx, yy = np.meshgrid(x, y)#快速生成坐标矩阵
for i,ax in enumerate(axs.reshape(-1)):
    m = Basemap(llcrnrlat=-60,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,ax=ax)
    m.pcolor(xx, yy, Uncertainties[i], cmap=cmap, latlon=True, norm=norm1)
    m.drawcoastlines()
    m.drawcountries()
    ax.set_title(labs[i],loc='left',fontweight='bold',fontsize = 20)
    # 去除边框
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()

cbar_ax = fig.add_axes([0.045, 0, 0.92, 0.02])

cb = matplotlib.colorbar.ColorbarBase(cbar_ax, ticks=bounds,cmap=cmap, orientation='horizontal',norm=norm1)
cb.set_label('Relative standard deviation (%)')
cb.set_ticks(bounds)
cb.set_ticklabels(bounds)
# cbar_ax.set_xticklabels(bounds)
plt.tight_layout()


# In[47]:


fig.savefig(r'E:/li/文章/审稿人意见/李润桐-NF-0810/figs/uncertainty.png',dpi=600,bbox_inches='tight')


# In[50]:


CH4_df.describe()


# In[48]:


rice


# # 计算总量和95%置信区间

# In[57]:


yield_df2 = pd.read_excel('simulation_results2.xlsx', sheet_name='Yield')
N2O_df2 = pd.read_excel('simulation_results2.xlsx', sheet_name='N2O')
CH4_df2 = pd.read_excel('simulation_results2.xlsx', sheet_name='CH4')

yield_df = pd.read_excel('simulation_results.xlsx', sheet_name='Yield')
N2O_df = pd.read_excel('simulation_results.xlsx', sheet_name='N2O')
CH4_df = pd.read_excel('simulation_results.xlsx', sheet_name='CH4')


# In[58]:


yield_df2


# In[68]:


def total_with_uncertainty(rice,index):
    start =time.time()
    rice['CH4_xgb_per_area'] = CH4_df[index]
    rice['CH4_xgb_per_area2'] = CH4_df2[index]
    rice.loc[rice['duration2'] == 0, 'CH4_xgb'] = rice['CH4_xgb_per_area']*rice['area']
    rice.loc[rice['duration2'] != 0, 'CH4_xgb'] = rice['CH4_xgb_per_area']*rice['physical_area']+ rice['CH4_xgb_per_area2']*(rice['area']-rice['physical_area'])
#     rice['CH4_xgb'] = rice['CH4_xgb_per_area']*rice['area']
    
    rice['N2O_xgb_per_area'] = N2O_df[index]
    rice['N2O_xgb_per_area2'] = N2O_df2[index]
    rice.loc[rice['duration2'] == 0, 'N2O_xgb'] = rice['N2O_xgb_per_area']*rice['area']
    rice.loc[rice['duration2'] != 0, 'N2O_xgb'] = rice['N2O_xgb_per_area']*rice['physical_area']+rice['N2O_xgb_per_area2']*(rice['area']-rice['physical_area'])


    
    rice['yield_xgb_per_area'] = yield_df[index]
    rice['yield_xgb_per_area2'] = yield_df2[index]
    rice.loc[rice['duration2'] == 0, 'yield_xgb'] = rice['yield_xgb_per_area']*rice['area']
    rice.loc[rice['duration2'] != 0, 'yield_xgb'] = rice['yield_xgb_per_area']*rice['physical_area']+rice['yield_xgb_per_area2']*(rice['area']-rice['physical_area'])
    
    CH4_total = sum(rice['CH4_xgb'])
    N2O_total = sum(rice['N2O_xgb'])
    yield_total = sum(rice['yield_xgb'])
    
    CH4_ave = CH4_total/sum(rice['area'])
    N2O_ave = N2O_total/sum(rice['area'])
    yield_ave = yield_total/sum(rice['area'])


    end = time.time()
    print('Running time: %s Seconds'%(end-start))
    print(index)
    print("Area-weighted average:")
    print(CH4_ave, N2O_ave, yield_ave)
    print("Total:")
    print(CH4_total, N2O_total, yield_total)


# In[69]:


index = 'Mean'
total_with_uncertainty(rice,index)


# In[70]:


index = 'Lower_CI'
total_with_uncertainty(rice,index)


# In[71]:


index = 'Upper_CI'
total_with_uncertainty(rice,index)

