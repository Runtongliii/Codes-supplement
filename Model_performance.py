#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from pandas.plotting import scatter_matrix   #绘制数值属性之间的相关性
from sklearn.impute import SimpleImputer     #处理缺失值
from sklearn.preprocessing import OrdinalEncoder    #处理文本属性，将类别转成数字
from sklearn.preprocessing import OneHotEncoder     #将类别值转化为独热向量
from sklearn.base import BaseEstimator, TransformerMixin   
#自定义转换器添加组合后的属性，前者构造函数时避免*args和**kargs，后者得到fit_transform()方法
from sklearn.pipeline import Pipeline    #构造流水线使数据转换的步骤以正确的顺序进行
from sklearn.preprocessing import StandardScaler    #标准化的转换器
from sklearn.compose import ColumnTransformer       #需要对不同的列组(数值列、分类列)应用不同的transformer   
from sklearn.metrics import mean_squared_error    #mse均方误差，rmse均方根误差
from sklearn.linear_model import LinearRegression   #线性回归模型
from sklearn.tree import DecisionTreeRegressor    #决策树
from sklearn.model_selection import GridSearchCV    #网格搜索法
from sklearn.model_selection import RandomizedSearchCV    #随机搜索法
from sklearn.ensemble import RandomForestRegressor    #随机森林
from sklearn.model_selection import cross_val_score    #交叉验证
from sklearn import metrics
import joblib
import json
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit#分层抽样
# from keras.wrappers.scikit_learn import KerasRegressor
from pandas import DataFrame as df
import time
import xgboost
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import sklearn


# In[52]:


CH4 = pd.read_csv('CH4.csv', index_col=0) 
N2O = pd.read_csv('N2O.csv', index_col=0) 
yields = pd.read_csv('yields.csv', index_col=0) 


# In[53]:


attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']
num_attr = ['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin']


# In[54]:


xgbr_CH4 = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=400)
xgbr_N2O = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=200)
xgbr_yield = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=500)


# In[55]:


scaler_CH4 = joblib.load('scaler_CH4.pkl')
scaler_N2O = joblib.load('scaler_N2O.pkl')
scaler_yield = joblib.load('scaler_yield.pkl')


# In[56]:


config = {
    "font.family":'serif', # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 15, # medium/large/small
    'font.style':'normal', # normal/italic/oblique
    'font.weight':'normal', # bold
    "mathtext.fontset":'cm',# 'cm' (Computer Modern)
    "font.serif": ['Arial'], # 'Simsun'宋体
    "axes.unicode_minus": False,# 用来正常显示负号
}
plt.rcParams.update(config)


# In[60]:


data=[CH4,N2O,yields]


# In[63]:


import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

fig=plt.figure(figsize=(15,4))
##rect可以设置子图的位置与大小
rect1 = [0, 0.95, 0.25, 0.95] 
# [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例），左下角为(0,0),右上角为(1,1)
rect2 = [0.33, 0.95, 0.25, 0.95]
rect3 = [0.66, 0.95, 0.25, 0.95]
rect=[rect1,rect2,rect3]
data=[CH4,N2O,yields]
labels=['CH4','N2O','yield']
models=[xgbr_CH4,xgbr_N2O,xgbr_yield]
xlabels=['Detected C$\mathregular{H_4}$ emissions (kg ha$\mathregular{^{-1}}$)',
        'Detected $\mathregular{N_2}$O emissions (kgN ha$\mathregular{^{-1}}$)',
        'Detected yield (t ha$\mathregular{^{-1}}$)']
ylabels=['Predicted C$\mathregular{H_4}$ emissions (kg ha$\mathregular{^{-1}}$)',
        'Predicted $\mathregular{N_2}$O emissions (kgN ha$\mathregular{^{-1}}$)',
        'Predicted yield (t ha$\mathregular{^{-1}}$)']
scalers = [scaler_CH4,scaler_N2O,scaler_yield]

for i in range(3):
    ax=plt.axes(rect[i])
    if i==0:
        x_train, x_test, y_train, y_test = train_test_split(data[i][attributes],data[i][labels[i]], test_size=0.2,random_state=0)
    else:
        x_train, x_test, y_train, y_test = train_test_split(data[i][attributes],data[i][labels[i]], test_size=0.2,random_state=42)
    
    
    scaler = scalers[i]
    x_train[num_attr] = scaler.transform(x_train[num_attr])
    x_test[num_attr] = scaler.transform(x_test[num_attr])
    models[i].fit(x_train, y_train)
    train_true=y_train
    test_true=y_test
    
    train_pred=models[i].predict(x_train)
    test_pred=models[i].predict(x_test)
    

    ax.scatter(train_true,train_pred,color='c',marker='h',label='Train')
    ax.scatter(test_true,test_pred,color='orange',marker='^',label='Test')
    
    if(i==0):
        ax.set_xlim(0,800)
        ax.set_ylim(0,800)
        ax.set_xticks([0,200,400,600,800])
        ax.set_yticks([0,200,400,600,800])
        ax.plot([0,800], [0,800],color='g',linewidth=2)
        train_rmse = 49.3
        ax.plot([0,800], [train_rmse, 800 + train_rmse], color='g', linestyle='--')
        ax.plot([0,800], [-train_rmse, 800 - train_rmse], color='g', linestyle='--')
        ax.set_title('a',loc='left',fontweight='bold',fontsize = 30)
    elif(i==1):
        ax.set_xlim(0,2.5)
        ax.set_ylim(0,2.5)
        ax.set_xticks([0,0.5,1,1.5,2,2.5])
        ax.set_yticks([0,0.5,1,1.5,2,2.5])
        ax.plot([0,2.5], [0,2.5],color='g',linewidth=2)
        train_rmse = 0.17
        ax.plot([0,2.5], [train_rmse, 2.5 + train_rmse], color='g', linestyle='--')
        ax.plot([0,2.5], [-train_rmse, 2.5 - train_rmse], color='g', linestyle='--')
        
        ax.set_title('b',loc='left',fontweight='bold',fontsize = 30)
    elif(i==2):
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        ax.set_xticks([0,2.5,5,7.5,10])
        ax.set_yticks([0,2.5,5,7.5,10])
        ax.plot([0,10], [0,10],color='g',linewidth=2)
        train_rmse = 0.43
        ax.plot([0,10], [train_rmse, 10 + train_rmse], color='g', linestyle='--', label='+RMSE Line')
        ax.plot([0,10], [-train_rmse, 10 - train_rmse], color='g', linestyle='--', label='-RMSE Line')
        ax.set_title('c',loc='left',fontweight='bold',fontsize = 30)

    ax.set_xlabel(xlabels[i])
    ax.set_ylabel(ylabels[i])
    ax.legend(loc=2,frameon=False,markerfirst=True) 
    
    # 计算R²和p值
    train_r2 = r2_score(train_true, train_pred)
    test_r2 = r2_score(test_true, test_pred)
    train_p_value = pearsonr(train_true, train_pred)[1]
    test_p_value = pearsonr(test_true, test_pred)[1]
    
    # 输出RMSE和MAE
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # 计算训练集的RMSE和MAE
    train_rmse = mean_squared_error(train_true, train_pred, squared=False)
    train_mae = mean_absolute_error(train_true, train_pred)

    # 计算测试集的RMSE和MAE
    test_rmse = mean_squared_error(test_true, test_pred, squared=False)
    test_mae = mean_absolute_error(test_true, test_pred)
    print(labels[i])
    print("训练集RMSE:", train_rmse)
    print("训练集MAE:", train_mae)
    print("测试集RMSE:", test_rmse)
    print("测试集MAE:", test_mae)
    
    
    # 添加R²和p值到图形中
    ax.text(0.98, 0.02, f'Train R²: {train_r2:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right')
    ax.text(0.98, 0.12, f'Test R²: {test_r2:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right')
#     ax.text(0.98, 0.22, f'p<0.001', transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right')

plt.tight_layout()
plt.show()


# In[12]:


fig.savefig(r'E:\li\文章\审稿人意见\李润桐-NF-0810\figs\scatter.png',dpi=600,bbox_inches='tight')


# # 残差图

# In[110]:


import matplotlib.pyplot as plt

# 加载模型
rf_CH4 = joblib.load('rf_CH4.pkl')
rf_N2O = joblib.load('rf_N2O.pkl')
rf_yield = joblib.load('rf_yield.pkl')

xgbr_CH4 = joblib.load('xgbr_CH4.pkl')
xgbr_N2O = joblib.load('xgbr_N2O.pkl')
xgbr_yield = joblib.load('xgbr_yield.pkl')

svr_CH4 = joblib.load('svr_CH4.pkl')
svr_N2O = joblib.load('svr_N2O.pkl')
svr_yield = joblib.load('svr_yield.pkl')

mlp_CH4 = joblib.load('mlp_CH4.pkl')
mlp_N2O = joblib.load('mlp_N2O.pkl')
mlp_yield = joblib.load('mlp_yield.pkl')


# In[120]:


len(data)


# In[125]:





# In[126]:


titles


# In[134]:


# 准备要绘制的图的数据
models = [rf_CH4, xgbr_CH4, svr_CH4, mlp_CH4, rf_N2O, xgbr_N2O, svr_N2O, mlp_N2O, rf_yield, xgbr_yield, svr_yield, mlp_yield]
model_names = ['RF', 'XGB', 'SVR', 'MLP']
output_names = ['CH4', 'N2O', 'Yield']
labs = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l']
titles = ["C$\mathregular{H_4}$","$\mathregular{N_2}$O", "Yield"]

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 11))

# 循环绘制每个图
for i, output in enumerate(output_names): #0,1,2
#     print(i)
    for j, model_name in enumerate(model_names): #0,1,2,3
#         print(j)
        idx = i * len(model_names) + j  # Calculate the correct index for the dataset
        print(idx)
        if idx < len(models):
            ax = axes[i, j]  # 此处调整为 (i, j)，而不是 (j, i)，以正确选择子图

            # 获取当前模型
            model = models[idx]

            # 此处添加模型的训练和预测代码，并绘制残差图
            x_train, x_test, y_train, y_test = train_test_split(data[i][attributes], data[i][labels[i]],
                                                        test_size=0.2, random_state=0)
            scaler = scalers[i]
            x_train[num_attr] = scaler.transform(x_train[num_attr])
            x_test[num_attr] = scaler.transform(x_test[num_attr])
            models[idx].fit(x_train, y_train)

            train_pred = models[idx].predict(x_train)
            test_pred = models[idx].predict(x_test)

            train_true = y_train
            test_true = y_test

            ax.scatter(train_true, train_true - train_pred, c='blue', marker='o', label='Training data')
            ax.scatter(test_true, test_true - test_pred, c='orange', marker='s', label='Test data')
            ax.axhline(y=0, color='gray', linestyle='--', lw=2)
            ax.set_xlabel('True values')
            ax.set_ylabel('Residuals')
#             ax.set_title('Residual plot for {}'.format(titles[j]))
            ax.set_title(label = labs[idx], loc='left',fontweight='bold',fontsize = 20)
            ax.legend(loc='best', fontsize='10', frameon=False)

#             # Calculate mean and standard deviation of residuals
#             train_residuals = train_true - train_pred
#             test_residuals = test_true - test_pred
#             train_residual_mean = np.mean(train_residuals)
#             train_residual_std = np.std(train_residuals)
#             test_residual_mean = np.mean(test_residuals)
#             test_residual_std = np.std(test_residuals)
#             ax.set_xlabel('True values')
#             ax.set_ylabel('Residuals')
#             ax.set_title('Residual plot for {} - {}'.format(output, model_name))
            ax.axhline(y=0, color='gray', linestyle='--', lw=2)

plt.tight_layout()
plt.show()


# In[135]:


fig.savefig("E:/li/文章/审稿人意见/李润桐-NF-0810/figs/residual_plot_total.jpg",dpi=600,bbox_inches='tight')


# In[136]:


# 准备要绘制的图的数据
models = [rf_CH4, xgbr_CH4, svr_CH4, mlp_CH4, rf_N2O, xgbr_N2O, svr_N2O, mlp_N2O, rf_yield, xgbr_yield, svr_yield, mlp_yield]
model_names = ['RF', 'XGB', 'SVR', 'MLP']
output_names = ['CH4', 'N2O', 'Yield']
labs = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l']
# titles = ["C$\mathregular{H_4}$","$\mathregular{N_2}$O", "Yield"]

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 11))

# 循环绘制每个图
for i, output in enumerate(output_names): #0,1,2
#     print(i)
    for j, model_name in enumerate(model_names): #0,1,2,3
#         print(j)
        idx = i * len(model_names) + j  # Calculate the correct index for the dataset
        print(idx)
        if idx < len(models):
            ax = axes[i, j]  # 此处调整为 (i, j)，而不是 (j, i)，以正确选择子图

            # 获取当前模型
            model = models[idx]

            # 此处添加模型的训练和预测代码，并绘制残差图
            x_train, x_test, y_train, y_test = train_test_split(data[i][attributes], data[i][labels[i]],
                                                                test_size=0.2, random_state=0)
            scaler = scalers[i]
            x_train[num_attr] = scaler.transform(x_train[num_attr])
            x_test[num_attr] = scaler.transform(x_test[num_attr])
            models[idx].fit(x_train, y_train)

            train_pred = models[idx].predict(x_train)
            test_pred = models[idx].predict(x_test)

            train_true = y_train
            test_true = y_test

            # Calculate mean and standard deviation of residuals
            train_residuals = train_true - train_pred
            test_residuals = test_true - test_pred

            # 绘制残差的密度直方图
            sns.histplot(train_residuals, kde=True, ax=ax, color='blue', label='Training residuals', stat='density')
            sns.histplot(test_residuals, kde=True, ax=ax, color='orange', label='Test residuals', stat='density')
            ax.set_xlabel('Residuals')  # 设置横轴标签
            ax.set_ylabel('Relative Frequency')  # 设置纵轴标签为相对频率
#             ax.set_title('Residual plot for {}'.format(titles[idx]))  # 设置标题
            ax.set_title(label = labs[idx], loc='left',fontweight='bold',fontsize = 20)
        #     ax.legend(loc='best')  # 添加图例
            ax.legend(loc='best', fontsize='9', frameon=False)

plt.tight_layout()
plt.show()


# In[137]:


fig.savefig("E:/li/文章/审稿人意见/李润桐-NF-0810/figs/residual_frequency_total.jpg",dpi=600,bbox_inches='tight')


# In[ ]:





# In[57]:





# In[61]:


labels = ["CH4", "N2O", "yield"]
labs = ['a', 'b','c']
data=[CH4,N2O,yields]


# In[71]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))  # Adjusting figure size and layout

for j, ax in enumerate(axes):
    idx = j  # Calculate the index for the dataset

    x_train, x_test, y_train, y_test = train_test_split(data[idx][attributes], data[idx][labels[idx]],
                                                        test_size=0.2, random_state=0)
    scaler = scalers[idx]
    x_train[num_attr] = scaler.transform(x_train[num_attr])
    x_test[num_attr] = scaler.transform(x_test[num_attr])
    models[idx].fit(x_train, y_train)

    train_pred = models[idx].predict(x_train)
    test_pred = models[idx].predict(x_test)

    train_true = y_train
    test_true = y_test

    ax.scatter(train_true, train_true - train_pred, c='blue', marker='o', label='Training data')
    ax.scatter(test_true, test_true - test_pred, c='orange', marker='s', label='Test data')
    ax.axhline(y=0, color='gray', linestyle='--', lw=2)
    ax.set_xlabel('True values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual plot for {}'.format(titles[idx]))
    ax.set_title(label = labs[idx], loc='left',fontweight='bold',fontsize = 20)
    ax.legend(loc='best', fontsize='10', frameon=False)

    # Calculate mean and standard deviation of residuals
    train_residuals = train_true - train_pred
    test_residuals = test_true - test_pred
    train_residual_mean = np.mean(train_residuals)
    train_residual_std = np.std(train_residuals)
    test_residual_mean = np.mean(test_residuals)
    test_residual_std = np.std(test_residuals)

    # Output residual statistics for training and testing sets
    print(f"{labels[idx]} Training set residual mean: {train_residual_mean}, Training set residual std: {train_residual_std}")
    print(f"{labels[idx]} Testing set residual mean: {test_residual_mean}, Testing set residual std: {test_residual_std}")

plt.tight_layout()
plt.show()


# In[73]:


fig.savefig("E:/li/文章/审稿人意见/李润桐-NF-0810/figs/residual_plot.jpg",dpi=600,bbox_inches='tight')


# In[104]:


models


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))  # Adjusting figure size and layout

for j, ax in enumerate(axes):
    idx = j  # Calculate the index for the dataset

    x_train, x_test, y_train, y_test = train_test_split(data[idx][attributes], data[idx][labels[idx]],
                                                        test_size=0.2, random_state=0)
    scaler = scalers[idx]
    x_train[num_attr] = scaler.transform(x_train[num_attr])
    x_test[num_attr] = scaler.transform(x_test[num_attr])
    models[idx].fit(x_train, y_train)

    train_pred = models[idx].predict(x_train)
    test_pred = models[idx].predict(x_test)

    train_true = y_train
    test_true = y_test

    ax.scatter(train_true, train_true - train_pred, c='blue', marker='o', label='Training data')
    ax.scatter(test_true, test_true - test_pred, c='orange', marker='s', label='Test data')
    ax.axhline(y=0, color='gray', linestyle='--', lw=2)
    ax.set_xlabel('True values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual plot for {}'.format(titles[idx]))
    ax.set_title(label = labs[idx], loc='left',fontweight='bold',fontsize = 20)
    ax.legend(loc='best', fontsize='10', frameon=False)


plt.tight_layout()
plt.show()


# In[75]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))  # Adjusting figure size and layout

for j, ax in enumerate(axes):
    idx = j  # Calculate the index for the dataset
    x_train, x_test, y_train, y_test = train_test_split(data[idx][attributes], data[idx][labels[idx]],
                                                        test_size=0.2, random_state=0)
    scaler = scalers[idx]
    x_train[num_attr] = scaler.transform(x_train[num_attr])
    x_test[num_attr] = scaler.transform(x_test[num_attr])
    models[idx].fit(x_train, y_train)

    train_pred = models[idx].predict(x_train)
    test_pred = models[idx].predict(x_test)

    train_true = y_train
    test_true = y_test

    # Calculate mean and standard deviation of residuals
    train_residuals = train_true - train_pred
    test_residuals = test_true - test_pred

    # 绘制残差的密度直方图
    sns.histplot(train_residuals, kde=True, ax=ax, color='blue', label='Training residuals', stat='density')
    sns.histplot(test_residuals, kde=True, ax=ax, color='orange', label='Test residuals', stat='density')
    ax.set_xlabel('Residuals')  # 设置横轴标签
    ax.set_ylabel('Relative Frequency')  # 设置纵轴标签为相对频率
    ax.set_title('Residual plot for {}'.format(titles[idx]))  # 设置标题
    ax.set_title(label = labs[idx], loc='left',fontweight='bold',fontsize = 20)
#     ax.legend(loc='best')  # 添加图例
    ax.legend(loc='best', fontsize='9', frameon=False)

    #
plt.tight_layout()
plt.show()


# In[76]:


fig.savefig("E:/li/文章/审稿人意见/李润桐-NF-0810/figs/residual_frequency.jpg",dpi=600,bbox_inches='tight')


# In[ ]:





# # 10-fold boxplot

# In[9]:


rf_CH4 = joblib.load('rf_CH4.pkl')
rf_N2O = joblib.load('rf_N2O.pkl')
rf_yield = joblib.load('rf_yield.pkl')

xgbr_CH4 = joblib.load('xgbr_CH4.pkl')
xgbr_N2O = joblib.load('xgbr_N2O.pkl')
xgbr_yield = joblib.load('xgbr_yield.pkl')

svr_CH4 = joblib.load('svr_CH4.pkl')
svr_N2O = joblib.load('svr_N2O.pkl')
svr_yield = joblib.load('svr_yield.pkl')

mlp_CH4 = joblib.load('mlp_CH4.pkl')
mlp_N2O = joblib.load('mlp_N2O.pkl')
mlp_yield = joblib.load('mlp_yield.pkl')


# In[17]:


CH4 = pd.read_csv('CH4.csv', index_col=0) 
N2O = pd.read_csv('N2O.csv', index_col=0) 
yields = pd.read_csv('yields.csv', index_col=0) 
CH4[num_attr] = scaler_CH4.transform(CH4[num_attr])
N2O[num_attr] = scaler_N2O.transform(N2O[num_attr])
yields[num_attr] = scaler_yield.transform(yields[num_attr])


# In[11]:


attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']
num_attr = ['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin']


# In[12]:


scaler_CH4 = joblib.load('scaler_CH4.pkl')
scaler_N2O = joblib.load('scaler_N2O.pkl')
scaler_yield = joblib.load('scaler_yield.pkl')


# In[26]:


CH4


# In[27]:


def results_10folds(model,model_name):#cross_validate是sklearn模块的函数，keras模型直接调用时出错了
    cv=KFold(n_splits=10,shuffle=True,random_state=42)
    r2=cross_validate(model,CH4[attributes], CH4["CH4"],cv=cv,scoring="r2",return_train_score=True,
                        verbose=True,n_jobs=-1)
    rmse=cross_validate(model,CH4[attributes], CH4["CH4"],cv=cv,scoring="neg_root_mean_squared_error",return_train_score=True,
                        verbose=True,n_jobs=-1)
#     test_rmse,test_r2=evaluate_model(model,x_test,y_test)
    CH4_R2_train[model_name]=r2["train_score"]
    CH4_RMSE_train[model_name]=-rmse["train_score"]
    CH4_R2_test[model_name]=r2["test_score"]
    CH4_RMSE_test[model_name]=-rmse["test_score"]
#     test_results[model_name]=[test_rmse,test_r2]

CH4_R2_train=pd.DataFrame()
CH4_R2_test=pd.DataFrame()
CH4_RMSE_train=pd.DataFrame()
CH4_RMSE_test=pd.DataFrame()

results_10folds(rf_CH4,"rf")
results_10folds(xgbr_CH4,"xgbr")
results_10folds(svr_CH4,"svr")
results_10folds(mlp_CH4,"mlp")


# In[45]:


CH4_R2_test.describe()


# In[37]:


CH4_RMSE_test.describe()


# In[38]:


def results_10folds(model,model_name):#cross_validate是sklearn模块的函数，keras模型直接调用时出错了
    cv=KFold(n_splits=10,shuffle=True,random_state=42)
    r2=cross_validate(model,N2O[attributes], N2O["N2O"],cv=cv,scoring="r2",return_train_score=True,
                        verbose=True,n_jobs=-1)
    rmse=cross_validate(model,N2O[attributes], N2O["N2O"],cv=cv,scoring="neg_root_mean_squared_error",return_train_score=True,
                        verbose=True,n_jobs=-1)
#     test_rmse,test_r2=evaluate_model(model,x_test,y_test)
    N2O_R2_train[model_name]=r2["train_score"]
    N2O_RMSE_train[model_name]=-rmse["train_score"]
    N2O_R2_test[model_name]=r2["test_score"]
    N2O_RMSE_test[model_name]=-rmse["test_score"]
#     test_results[model_name]=[test_rmse,test_r2]

N2O_R2_train=pd.DataFrame()
N2O_R2_test=pd.DataFrame()
N2O_RMSE_train=pd.DataFrame()
N2O_RMSE_test=pd.DataFrame()

results_10folds(rf_N2O,"rf")
results_10folds(xgbr_N2O,"xgbr")
results_10folds(svr_N2O,"svr")
results_10folds(mlp_N2O,"mlp")


# In[39]:


N2O_R2_test.describe()


# In[44]:


N2O_RMSE_test.describe()


# In[41]:


def results_10folds(model,model_name):#cross_validate是sklearn模块的函数，keras模型直接调用时出错了
    cv=KFold(n_splits=10,shuffle=True,random_state=42)
    r2=cross_validate(model,yields[attributes], yields["yield"],cv=cv,scoring="r2",return_train_score=True,
                        verbose=True,n_jobs=-1)
    rmse=cross_validate(model,yields[attributes], yields["yield"],cv=cv,scoring="neg_root_mean_squared_error",return_train_score=True,
                        verbose=True,n_jobs=-1)
#     test_rmse,test_r2=evaluate_model(model,x_test,y_test)
    yield_R2_train[model_name]=r2["train_score"]
    yield_RMSE_train[model_name]=-rmse["train_score"]
    yield_R2_test[model_name]=r2["test_score"]
    yield_RMSE_test[model_name]=-rmse["test_score"]
#     test_results[model_name]=[test_rmse,test_r2]

yield_R2_train=pd.DataFrame()
yield_R2_test=pd.DataFrame()
yield_RMSE_train=pd.DataFrame()
yield_RMSE_test=pd.DataFrame()

results_10folds(rf_yield,"rf")
results_10folds(xgbr_yield,"xgbr")
results_10folds(svr_yield,"svr")
results_10folds(mlp_yield,"mlp")


# In[79]:


yield_R2_test


# In[90]:


# 更改列名为RF, XGB, SVR, MLP
yield_RMSE_test.columns = ['RF', 'XGB', 'SVR', 'MLP']
yield_R2_test.columns = ['RF', 'XGB', 'SVR', 'MLP']
CH4_R2_test.columns = ['RF', 'XGB', 'SVR', 'MLP']
CH4_RMSE_test.columns = ['RF', 'XGB', 'SVR', 'MLP']
N2O_R2_test.columns = ['RF', 'XGB', 'SVR', 'MLP']
N2O_RMSE_test.columns = ['RF', 'XGB', 'SVR', 'MLP']


# In[91]:


yield_RMSE_test
yield_R2_test

CH4_R2_test
CH4_RMSE_test

N2O_R2_test
N2O_RMSE_test


# In[99]:


# 创建绘图
fig  = plt.figure(figsize=(12, 6))

# 绘制箱线图和数据点(RMSE)
plt.subplot(2, 3, 1)
sns.boxplot(data=CH4_RMSE_test,showmeans=True)  # 绘制箱线图
sns.swarmplot(data=CH4_RMSE_test, color='black', alpha=0.5, size=5)  # 添加数据点
plt.title('a',loc='left',fontweight='bold')
plt.ylabel('RMSE (kg ha$^{-1}$)')

plt.subplot(2, 3, 2)
sns.boxplot(data=N2O_RMSE_test,showmeans=True)  # 绘制箱线图
sns.swarmplot(data=N2O_RMSE_test, color='black', alpha=0.5, size=5)  # 添加数据点
plt.title('b',loc='left',fontweight='bold')
plt.ylabel('RMSE (kg N ha$^{-1}$)')

plt.subplot(2, 3, 3)
sns.boxplot(data=yield_RMSE_test,showmeans=True)  # 绘制箱线图
sns.swarmplot(data=yield_RMSE_test, color='black', alpha=0.5, size=5)  # 添加数据点
plt.title('c',loc='left',fontweight='bold')
plt.ylabel('RMSE (kg ha$^{-1}$)')




# 绘制箱线图和数据点(R2)
plt.subplot(2, 3, 4)
sns.boxplot(data=CH4_R2_test,  showmeans=True)  # 绘制箱线图
sns.swarmplot(data=CH4_R2_test, color='black', alpha=0.5, size=5)  # 添加数据点
plt.title('d',loc='left',fontweight='bold')
plt.ylabel('R$^2$')

plt.subplot(2, 3, 5)
sns.boxplot(data=N2O_R2_test,  showmeans=True)  # 绘制箱线图
sns.swarmplot(data=N2O_R2_test, color='black', alpha=0.5, size=5)  # 添加数据点
plt.title('e',loc='left',fontweight='bold')
plt.ylabel('R$^2$')

plt.subplot(2, 3, 6)
sns.boxplot(data=yield_R2_test,  showmeans=True)  # 绘制箱线图
sns.swarmplot(data=yield_R2_test, color='black', alpha=0.5, size=5)  # 添加数据点
plt.title('f',loc='left',fontweight='bold')
plt.ylabel('R$^2$')

plt.tight_layout()  # 调整布局以防止重叠
plt.show()


# In[101]:


fig.savefig(r"E:\li\文章\审稿人意见\李润桐-NF-0810\figs\10-fold.png")


# In[83]:


# 绘制箱线图并添加数据点
plt.figure(figsize=(10, 6))
sns.boxplot(data=yield_RMSE_test)  # 绘制箱线图
sns.swarmplot(data=yield_RMSE_test, color='black', alpha=0.5, size=5)  # 添加数据点
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.show()

