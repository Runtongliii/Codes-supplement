#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
import time


# In[3]:


CH4 = pd.read_csv('CH4.csv', index_col=0) 
N2O = pd.read_csv('N2O.csv', index_col=0) 
yields = pd.read_csv('yields.csv', index_col=0) 


# In[4]:


N2O


# # 建立测试集

# In[158]:


attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']


# In[159]:


x_train, x_test, y_train, y_test = train_test_split(N2O[attributes], N2O["N2O"], test_size=0.2,random_state=42)


# # Standardization

# In[160]:


from sklearn.preprocessing import StandardScaler

# 初始化标准化处理器
scaler_N2O = StandardScaler()
num_attr = ['duration', 'temp', 'prec', 'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin']
# 对CH4数据集的前10列进行标准化处理
x_train[num_attr] = scaler_N2O.fit_transform(x_train[num_attr])

joblib.dump(scaler_N2O, 'scaler_N2O.pkl')


# In[162]:


x_test


# In[163]:


x_test[num_attr] = scaler_N2O.transform(x_test[num_attr])


# # Hyperparameter optimization
# Bayesian, grid and random search

# In[10]:


#模型指标
def evaluate_model(model,x,y):
    pred=model.predict(x)
    rmse=np.sqrt(mean_squared_error(y,pred))
    r2=metrics.r2_score(y,pred)
    return rmse,r2


# In[11]:


# grid search


# In[110]:


start =time.time()
param_grid=[
    {'n_estimators':[50,100,110,120,130,140,150,200],
#      'max_features':['auto','sqrt','log2'],
     'max_features':[3,6,9,12],
     'max_depth':[5,10,15,20,25]}
]
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg,param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True,verbose=1,n_jobs=-1,refit=True)
#refit=True，一旦通过交叉验证找到了最佳估算器，将在整个训练集上重新训练，提供更多的数据很可能提升性能
grid_search.fit(x_train,y_train)
end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[111]:


grid_search.best_params_


# In[112]:


rf= grid_search.best_estimator_
rf.fit(x_train,y_train)


# In[113]:


print(evaluate_model(rf,x_train,y_train))


# In[114]:


print(evaluate_model(rf,x_test,y_test))


# In[115]:


# 训练集交叉验证
cv = cross_val_score(rf,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[18]:


# random search


# In[122]:


from sklearn.model_selection import RandomizedSearchCV

# 设置参数空间
param_dist = {
    'n_estimators': np.arange(50, 201, 10),  # 从50到200中每隔10取一个数
    'max_features': np.arange(3, 15, 2),      # 从3到12中每隔3取一个数
    'max_depth': np.arange(5, 26, 5)          # 从5到25中每隔5取一个数
}

start = time.time()

# 初始化随机搜索
forest_reg = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_dist,
    n_iter=100,  # 设置搜索次数
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

# 进行随机搜索
random_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[123]:


random_search.best_params_


# In[124]:


rf= random_search.best_estimator_
rf.fit(x_train,y_train)
print(evaluate_model(rf,x_train,y_train))
print(evaluate_model(rf,x_test,y_test))
# 训练集交叉验证
cv = cross_val_score(rf,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[22]:


# Bayesian Search 


# In[119]:


from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
import time

# 设置参数空间
param_dist = {
    'n_estimators': Integer(50, 200),   # 设置n_estimators为整数范围
    'max_features': Integer(3, 12),      # 设置max_features为整数范围
    'max_depth': Integer(5, 25)          # 设置max_depth为整数范围
}

start = time.time()

# 初始化贝叶斯优化
forest_reg = RandomForestRegressor(random_state=42)
bayes_search = BayesSearchCV(
    forest_reg,
    search_spaces=param_dist,
    n_iter=50,  # 设置搜索次数
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

# 进行贝叶斯优化
bayes_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[120]:


bayes_search.best_params_


# In[121]:


rf= bayes_search.best_estimator_
rf.fit(x_train,y_train)
print(evaluate_model(rf,x_train,y_train))
print(evaluate_model(rf,x_test,y_test))
# 训练集交叉验证
cv = cross_val_score(rf,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[ ]:





# # XGB

# In[125]:


start =time.time()
x=XGBRegressor(random_state=42)
xgbr_param_grid=[
    {'n_estimators':[50,100,200,300,400,500],
     'learning_rate':[0.1,0.2,0.3,0.4],
     'max_depth':[2,4,6,8,10]}]
xgbr_grid_search=GridSearchCV(x,xgbr_param_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True,verbose=1,n_jobs=-1,refit=True)
#记得refit=True
xgbr_grid_search.fit(x_train,y_train)


# In[126]:


xgbr_grid_search.best_params_


# In[127]:


xgbr=xgbr_grid_search.best_estimator_
xgbr.fit(x_train,y_train)


# In[128]:


print(evaluate_model(xgbr,x_test,y_test))
print(evaluate_model(xgbr,x_train,y_train))
# 训练集交叉验证
cv = cross_val_score(xgbr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[30]:


# random search


# In[129]:


# 设置参数空间
xgbr_param_dist = {
    'n_estimators': np.arange(50, 501, 50),
    'learning_rate': np.arange(0.1, 0.5, 0.1),
    'max_depth': np.arange(2, 11, 2)
}

start = time.time()

# 初始化随机搜索
xgbr = XGBRegressor(random_state=42)
xgbr_random_search = RandomizedSearchCV(
    xgbr,
    param_distributions=xgbr_param_dist,
    n_iter=50,  # 设置搜索次数
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

# 进行随机搜索
xgbr_random_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[130]:


print(xgbr_random_search.best_params_)


# In[131]:


xgbr=xgbr_random_search.best_estimator_
xgbr.fit(x_train,y_train)


# In[132]:


print(evaluate_model(xgbr,x_test,y_test))
print(evaluate_model(xgbr,x_train,y_train))
# 训练集交叉验证
cv = cross_val_score(xgbr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[35]:


## Bayesian Search 


# In[133]:


from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBRegressor
import time

# 设置参数空间
xgbr_param_dist = {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(0.1, 0.5),
    'max_depth': Integer(2, 10)
}

start = time.time()

# 初始化贝叶斯优化
xgbr = XGBRegressor(random_state=42)
xgbr_bayes_search = BayesSearchCV(
    xgbr,
    search_spaces=xgbr_param_dist,
    n_iter=50,  # 设置搜索次数
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

# 进行贝叶斯优化
xgbr_bayes_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[134]:


xgbr_bayes_search.best_params_


# In[135]:


xgbr= xgbr_bayes_search.best_estimator_
xgbr.fit(x_train,y_train)
print(evaluate_model(xgbr,x_train,y_train))
print(evaluate_model(xgbr,x_test,y_test))
# 训练集交叉验证
cv = cross_val_score(xgbr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[ ]:





# In[ ]:





# # SVR

# In[39]:


from sklearn.svm import SVR


# In[101]:


# start =time.time()
# s=SVR(kernel="rbf") #C=10000也是网格搜索得到的
# param_grid=[
#     {'C':[10,100,1000,1500,2000],
#      "gamma":[0.1,0.2,0.3,0.4,0.5],
#      #'kernel':["rbf","sigmoid","poly"],
#     # 'epsilon':[40,45,50]
#     }
# ]
# #epsilon，预测值与实际值的距离在epsilon内，损失函数值为0，默认0.1
# #C，误差项的惩罚函数，一般为10的n次幂，C越大，希望松弛变量接近0，即对误分类的惩罚越大，容易过拟合；减小C正则化，默认值1
# #kernel：参数选择有RBF（高斯核）, Linear（线性核函数）, Poly（多项式核函数）, Sigmoid（sigmoid核函数）, 默认的是"RBF";
# grid_search=GridSearchCV(s,param_grid,cv=5,scoring='neg_mean_squared_error',
#                          return_train_score=True, verbose=True,n_jobs=-1,refit=True)

# end = time.time()
# print('Running time: %s Seconds' % (end - start))

start = time.time()

s = SVR(kernel="rbf")
param_grid = [
    {
        'C': [10, 100, 1000, 1500, 2000],
       "gamma": [0.1, 0.2, 0.3, 0.4, 0.5],
#         'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5],
#         'kernel': ['rbf', 'sigmoid', 'poly']
    }
]

grid_search = GridSearchCV(s, param_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True, verbose=True, n_jobs=-1, refit=True)
grid_search.fit(x_train,y_train)
end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[102]:


grid_search.best_params_


# In[103]:


svr = grid_search.best_estimator_
svr.fit(x_train,y_train)
print(evaluate_model(svr,x_test,y_test))
print(evaluate_model(svr,x_train,y_train))
cv = cross_val_score(svr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[43]:


# Random Search 


# In[104]:


start = time.time()

s = SVR(kernel="rbf")
param_dist = {
    'C': np.logspace(1, 4, 5),  # 设置C为对数空间
    'gamma': np.linspace(0.1, 0.5, 5),  # 设置gamma为线性空间
#     'epsilon': np.linspace(0.1, 0.5, 5),  # 设置epsilon为线性空间
    #'kernel': ['rbf', 'sigmoid', 'poly']
}

random_search = RandomizedSearchCV(
    s,
    param_distributions=param_dist,
    n_iter=50,  # 设置搜索次数
    cv=10,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

random_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[105]:


random_search.best_params_


# In[106]:


svr = random_search.best_estimator_
svr.fit(x_train,y_train)
print(evaluate_model(svr,x_test,y_test))
print(evaluate_model(svr,x_train,y_train))
cv = cross_val_score(svr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[46]:


# Bayesian Search 


# In[107]:


start = time.time()

s = SVR(kernel="rbf")
param_dist = {
    'C': Real(10, 2000),  # 设置C为连续空间
    'gamma': Real(0.1, 0.5),  # 设置gamma为连续空间
#     'epsilon': Real(0.1, 0.5),  # 设置epsilon为连续空间
#     'kernel': ['rbf', 'sigmoid', 'poly']
}

bayes_search = BayesSearchCV(
    s,
    search_spaces=param_dist,
    n_iter=50,  # 设置搜索次数
    cv=10,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

bayes_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[108]:


bayes_search.best_params_


# In[109]:


svr = bayes_search.best_estimator_
svr.fit(x_train,y_train)
print(evaluate_model(svr,x_test,y_test))
print(evaluate_model(svr,x_train,y_train))
cv = cross_val_score(svr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[ ]:





# In[ ]:





# # MLP

# In[82]:


from sklearn.neural_network import MLPRegressor


# In[136]:


mlp=MLPRegressor(early_stopping=True, hidden_layer_sizes=(1000,500),
             learning_rate_init=0.002, max_iter=500, random_state=42,
             verbose=True)


# In[ ]:





# In[ ]:





# In[137]:


start = time.time()

param_grid=[
    {'activation':['identity', 'logistic', 'tanh', 'relu'],
     'learning_rate': ['constant','adaptive'],
     'hidden_layer_sizes':[(100),(150),(200),(100,50),(200,100,50)],
      'learning_rate_init': [0.01,0.001,0.0001]
    }
]

grid_search=GridSearchCV(mlp,param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True,verbose=1,refit=True)
#refit=True，一旦通过交叉验证找到了最佳估算器，将在整个训练集上重新训练，提供更多的数据很可能提升性能
grid_search.fit(x_train,y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[138]:


print(grid_search.best_params_)


# In[139]:


mlp=grid_search.best_estimator_
mlp.fit(x_train,y_train)
print(evaluate_model(mlp,x_test,y_test))
print(evaluate_model(mlp,x_train,y_train))
cv = cross_val_score(mlp,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[54]:


# Random Search 


# In[140]:


start = time.time()

mlp = MLPRegressor(random_state=42)
param_dist = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'learning_rate': ['constant', 'adaptive'],
    'hidden_layer_sizes': [(100,), (150,), (200,), (100, 50), (200, 100, 50)],
    'learning_rate_init': [0.01,0.001,0.0001]
}

random_search = RandomizedSearchCV(
    mlp,
    param_distributions=param_dist,
    n_iter=50,  # 设置搜索次数
    cv=10,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

random_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[141]:


print(random_search.best_params_)


# In[142]:


mlp=random_search.best_estimator_
mlp.fit(x_train,y_train)
print(evaluate_model(mlp,x_test,y_test))
print(evaluate_model(mlp,x_train,y_train))
cv = cross_val_score(mlp,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[57]:


# Bayesian Search 


# In[143]:


from skopt.space import Categorical


# In[145]:


start = time.time()

mlp = MLPRegressor(random_state=42)
# 定义超参数空间
param_dist = {
    'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
    'learning_rate': Categorical(['constant', 'adaptive']),
    'hidden_layer_sizes': Integer(50, 300),  # 隐藏层神经元数量
     'learning_rate_init': Real(0.0001, 0.01, prior='log-uniform'),

}

bayes_search = BayesSearchCV(
    mlp,
    search_spaces=param_dist,
    n_iter=50,  # 设置搜索次数
    cv=10,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

bayes_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[146]:


print(bayes_search.best_params_)


# In[147]:


mlp=bayes_search.best_estimator_
mlp.fit(x_train,y_train)
print(evaluate_model(mlp,x_test,y_test))
print(evaluate_model(mlp,x_train,y_train))
cv = cross_val_score(mlp,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[149]:


y_test


# # 保存模型

# In[171]:


rf_N2O = RandomForestRegressor(max_depth=15, max_features=6, n_estimators=130)
xgbr_N2O= XGBRegressor(learning_rate=0.3, max_depth=4, n_estimators=50)
svr_N2O = SVR(C=10, gamma=0.2)
mlp_N2O = MLPRegressor(activation='relu', hidden_layer_sizes=(200, 100, 50), learning_rate='constant', learning_rate_init=0.01)


# In[172]:


rf_N2O.fit(x_train,y_train)
xgbr_N2O.fit(x_train,y_train)
svr_N2O.fit(x_train,y_train)
mlp_N2O.fit(x_train,y_train)


# In[173]:


y_train


# In[174]:


joblib.dump(rf_N2O,'rf_N2O.pkl')
joblib.dump(xgbr_N2O,'xgbr_N2O.pkl')
joblib.dump(svr_N2O,'svr_N2O.pkl')
joblib.dump(mlp_N2O,'mlp_N2O.pkl')

