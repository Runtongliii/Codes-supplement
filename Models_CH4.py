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


filename=r"E:\li\文章\审稿人意见\李润桐-NF-0810\李润桐-NF-0810\Supplementary Data.xlsx"
CH4 = pd.read_excel(filename, sheet_name="CH4",engine='openpyxl')
N2O = pd.read_excel(filename, sheet_name="N2O",engine='openpyxl')
yields = pd.read_excel(filename, sheet_name="Yield",engine='openpyxl')

# 使用 get_dummies() 函数将 "water_regime" 列转换为 one-hot 编码
CH4 = pd.get_dummies(CH4, columns=['water_regime'])
N2O = pd.get_dummies(N2O, columns=['water_regime'])
yields = pd.get_dummies(yields, columns=['water_regime'])

attr_CH4 = ['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
      'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed', 
       'CH4']
CH4 = CH4[attr_CH4]
 
attr_N2O = ['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
      'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed', 
       'N2O']
N2O=N2O[attr_N2O]

attr_yield = ['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed', 
       'yield']
yields = yields[attr_yield]


# In[5]:


# 异常值删除前的VIF
# multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    # 创建一个数据框来保存 VIF 值
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

# 对每个数据框应用 calculate_vif 函数
CH4_vif = calculate_vif(CH4)
N2O_vif = calculate_vif(N2O)
yields_vif = calculate_vif(yields)


# In[15]:


data = CH4
data = data.drop(data.columns[-1], axis=1)


# In[16]:


data


# In[17]:


# lh删除异常值方法
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator

# 1. 孤立森林
clf1 = IsolationForest(contamination=0.1)
outliers1 = clf1.fit_predict(data)

# 2. One-Class SVM
clf2 = OneClassSVM()
outliers2 = clf2.fit_predict(data)

# 3. DBSCAN
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
clf3 = DBSCAN(eps=0.5, min_samples=5)
outliers3 = clf3.fit_predict(data_scaled)

# 4. LOF
clf4 = LocalOutlierFactor()
outliers4 = clf4.fit_predict(data)

# 5. K均值
clf5 = KMeans(n_clusters=2, n_init=10)
outliers5 = clf5.fit_predict(data)

# 6. 高斯混合模型
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
clf6 = GaussianMixture(n_components=2)
outliers6 = clf6.fit_predict(data_scaled)

# 7. 自编码器
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
clf7 = PCA(n_components=1)
data_pca = clf7.fit_transform(data_scaled)
reconstructed_data = clf7.inverse_transform(data_pca)
reconstruction_error = np.mean((data_scaled - reconstructed_data) ** 2, axis=1)
threshold = np.percentile(reconstruction_error, 95)
outliers7 = (reconstruction_error < threshold).astype(int)

# 8. 随机投影
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
clf8 = PCA(n_components=1, random_state=0)
data_pca = clf8.fit_transform(data_scaled)
outliers8 = ((data_scaled - clf8.inverse_transform(data_pca)) ** 2).sum(axis=1)


outliers1 = np.where(outliers1 == -1, 1, 0)
outliers2 = np.where(outliers2 == -1, 1, 0)
outliers3 = np.where(outliers3 == -1, 1, 0)
outliers4 = np.where(outliers4 == -1, 1, 0)
outliers5 = np.where(outliers5 == 1, 0, 1)
outliers6 = np.where(outliers6 == 0, 1, 0)
outliers7 = np.where(outliers7 == 1, 0, 1)
outliers8 = np.where(outliers8 > np.percentile(outliers8, 95), 1, 0)


# 计算每个outliers中0和1的数量，并将数量多的分组修改为0，数量少的分组修改为1
def adjust_outliers(outliers):
    count_0 = np.sum(outliers == 0)
    count_1 = np.sum(outliers == 1)
    if count_0 < count_1:
        temp = outliers.copy()
        outliers[outliers == 0] = 1
        outliers[temp == 1] = 0
    return outliers

outliers1 = adjust_outliers(outliers1)
outliers2 = adjust_outliers(outliers2)
outliers3 = adjust_outliers(outliers3)
outliers4 = adjust_outliers(outliers4)
outliers5 = adjust_outliers(outliers5)
outliers6 = adjust_outliers(outliers6)
outlies7 = adjust_outliers(outliers7)
outliers8 = adjust_outliers(outliers8)

outliers = np.stack([outliers1, outliers2, outliers3, outliers4, outliers5, outliers6, outliers7, outliers8])
outliers = outliers.T
methods = ["Isolation Forest", "One-Class SVM", "DBSCAN", "LOF", "K Means", "Gaussian Mixture Model", "Autoencoder", "Random Projection"]
#outliers_list = [outliers1.T, outliers2.T, outliers3.T, outliers4.T, outliers5.T, outliers6.T, outliers7.T, outliers8.T]
outlier = pd.DataFrame(outliers)


# In[20]:


outlier


# In[30]:


# 假设CH4数据存储在名为CH4的DataFrame中，数据存储在名为data的DataFrame中
zero_rows = outlier[(outlier == 0).all(axis=1)]  # 选择全为零的行
CH4_zero_rows = CH4.loc[zero_rows.index]   # 根据选择的行索引选择CH4对应的行


# In[31]:


CH4_zero_rows = CH4_zero_rows.drop(CH4_zero_rows.columns[-1], axis=1)


# In[32]:


CH4_zero_rows


# In[33]:


calculate_vif(CH4_zero_rows)


# In[34]:


CH4


# # IQR remove outliers

# In[35]:


# IQR remove outliers
# 创建异常值处理函数
def replace_outliers_with_nan(df, attributes):
    df_clean = df.copy()
    for attribute in attributes:
        Q1 = df_clean[attribute].quantile(0.25)
        Q3 = df_clean[attribute].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 将异常值替换为 NaN
        df_clean.loc[(df_clean[attribute] < lower_bound) | (df_clean[attribute] > upper_bound), attribute] = None
    return df_clean

# 创建删除所有包含空值的行函数
def drop_rows_with_nan(df):
    return df.dropna()

# 指定连续型数值变量的属性名


# 对每个数据框应用异常值处理和删除空值行的函数
attributes = ['duration', 'temp', 'prec', 'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin','CH4']
CH4_clean = replace_outliers_with_nan(CH4, attributes)
CH4_clean = drop_rows_with_nan(CH4_clean)

attributes = ['duration', 'temp', 'prec', 'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin','N2O']
N2O_clean = replace_outliers_with_nan(N2O, attributes)
N2O_clean = drop_rows_with_nan(N2O_clean)

attributes = ['duration', 'temp', 'prec', 'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin','yield']
yields_clean = replace_outliers_with_nan(yields, attributes)
yields_clean = drop_rows_with_nan(yields_clean)

# 这里的 CH4_clean、N2O_clean 和 yields_clean 是处理后的数据框


# In[36]:


CH4_clean.describe()


# In[37]:


CH4_clean.describe()


# In[39]:


yields_clean.describe()


# In[10]:


# 保存去除异常值后的数据
CH4_clean.to_csv("CH4.csv")
N2O_clean.to_csv('N2O.csv')
yields_clean.to_csv('yields.csv')


# In[40]:


CH4 = pd.read_csv('CH4.csv', index_col=0) 
N2O = pd.read_csv('N2O.csv', index_col=0) 
yields = pd.read_csv('yields.csv', index_col=0) 


# In[41]:


CH4


# In[42]:


# multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    # 创建一个数据框来保存 VIF 值
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

CH4 = CH4.drop(CH4.columns[-1], axis=1)
N2O = N2O.drop(N2O.columns[-1], axis=1)
yields = yields.drop(yields.columns[-1], axis=1)

# 对每个数据框应用 calculate_vif 函数
CH4_vif = calculate_vif(CH4)
N2O_vif = calculate_vif(N2O)
yields_vif = calculate_vif(yields)

# 这里的 CH4_vif、N2O_vif 和 yields_vif 是每个特征的 VIF 值
# 先计算每个变量的vif值，再重复计算

# df = yields
# vif = calculate_vif(df.iloc[:,:-1])
# while (vif['VIF'] > 10).any():
#     remove = vif.sort_values(by='VIF',ascending=False)['feature'][:1].values[0]
#     df.drop(remove,axis=1,inplace=True)
#     vif = calculate_vif(df)


# In[43]:


CH4_vif


# In[44]:


N2O_vif


# In[45]:


yields_vif


# In[46]:



# 定义要保存的文件路径
output_file = r"E:\li\文章\审稿人意见\VIF.xlsx"

# 创建 Excel writer 对象
with pd.ExcelWriter(output_file) as writer:
    # 将 CH4_vif DataFrame 中的 VIF 值保存到 "VIF_CH4" sheet
    CH4_vif.to_excel(writer, sheet_name='VIF_CH4', index=False)
    
    # 将 N2O_vif DataFrame 中的 VIF 值保存到 "VIF_N2O" sheet
    N2O_vif.to_excel(writer, sheet_name='VIF_N2O', index=False)
    
    # 将 yields_vif DataFrame 中的 VIF 值保存到 "VIF_yield" sheet
    yields_vif.to_excel(writer, sheet_name='VIF_yield', index=False)


# # CEEMDAN

# In[19]:


# # 共线性问题

# from PyEMD import CEEMDAN  # 从PyEMD库中导入CEEMDAN方法，用于进行经验模态分解。
# from scipy.signal import periodogram  # 从SciPy库中导入periodogram函数，用于计算功率谱密度估计。
# csv_file = 'CH4.csv'  # 指定CSV文件路径和文件名。
# data = pd.read_csv(csv_file)  # 使用Pandas的read_csv函数读取CSV文件数据。
# time_series = data['CH4'].values  # 从数据中提取'A'列的数值作为时间序列数据。
# ceemdan = CEEMDAN()  # 创建CEEMDAN对象。
# ceemdan.extrema_detection = "parabol"  # 设置极值点检测方法为"parabol"。
# ceemdan.ceemdan(time_series)  # 对时间序列数据进行CEEMDAN分解。
# imfs, residue = ceemdan.get_imfs_and_residue()  # 获取分解得到的IMFs和残差。
# # 创建一个字典，将数据存储到 DataFrame 中
# data = {"IMF{}".format(i+1): imfs[i] for i in range(len(imfs))}
# data["Residue"] = residue
# # 将数据字典转换为 DataFrame
# df = pd.DataFrame(data)
# # 将 DataFrame 保存为 Excel 文件
# excel_file = "ceemdan_output.xlsx"
# df.to_excel(excel_file, index=False)
# print(f"数据已保存到 {excel_file}")


# In[20]:


# def plot_components(time_series, imfs, residue):
#     # 获取IMF分量的数量
#     n_imfs = imfs.shape[0]

#     # 创建具有适当大小的图形
#     plt.figure(figsize=(12, 2 * n_imfs))

#     # 生成时间点的数组
#     time_points = np.arange(0, len(time_series))

#     # 单独绘制每个IMF分量
#     for i in range(n_imfs):
#         plt.subplot(n_imfs + 1, 1, i + 1)  # 为每个IMF分量创建一个子图
#         plt.plot(time_points, imfs[i, :], label=f"IMF{i + 1}")  # 绘制IMF分量
#         plt.legend()  # 添加图例

#     # 绘制残差分量
#     plt.subplot(n_imfs + 1, 1, n_imfs + 1)  # 为残差分量创建一个子图
#     plt.plot(time_points, residue, label="残差")  # 绘制残差分量
#     plt.legend()  # 添加图例

#     # 调整子图之间的间距并显示图形
#     plt.tight_layout()
#     plt.show()


# In[21]:


# def plot_spectrum(imfs, residue):
#     # 获取IMF分量的数量
#     n_imfs = imfs.shape[0]

#     # 创建具有适当大小的图形
#     plt.figure(figsize=(10, 2 * n_imfs))

#     # 绘制每个IMF分量的频谱
#     for i in range(n_imfs):
#         plt.subplot(n_imfs + 1, 1, i + 1)  # 创建每个IMF分量的子图
#         f, p = periodogram(imfs[i, :], fs=1)  # 计算IMF分量的功率谱密度
#         plt.semilogy(f, p, label=f"IMF{i + 1} Spectrum")  # 绘制功率谱密度曲线
#         plt.xlabel("Frequency")  # 设置x轴标签为频率
#         plt.ylabel("Power")  # 设置y轴标签为功率
#         plt.legend()  # 添加图例

#     # 绘制残差分量的频谱
#     plt.subplot(n_imfs + 1, 1, n_imfs + 1)  # 创建残差分量的子图
#     f, p = periodogram(residue, fs=1)  # 计算残差分量的功率谱密度
#     plt.semilogy(f, p, label="Residue Spectrum")  # 绘制功率谱密度曲线
#     plt.xlabel("Frequency")  # 设置x轴标签为频率
#     plt.ylabel("Power")  # 设置y轴标签为功率
#     plt.legend()  # 添加图例

#     # 调整子图之间的间距并显示图形
#     plt.tight_layout()
#     plt.show()


# In[22]:


# import matplotlib.pyplot as plt

# # 遍历每个特征并绘制折线图
# for column in CH4.columns:
#     plt.figure()  # 创建一个新的图形
#     plt.plot(CH4[column])  # 绘制特征的折线图
#     plt.title(f'Line Plot of {column}')  # 添加标题
#     plt.xlabel('Index')  # 添加 x 轴标签
#     plt.ylabel('Value')  # 添加 y 轴标签
#     plt.show()  # 显示图形


# In[23]:


# # 绘制各个IMF分量和残差分量的时域图
# plot_components(time_series, imfs, residue)

# # 绘制各个IMF分量和残差分量的频谱图
# plot_spectrum(imfs, residue)
# def reconstruct_signal(imfs, residue):
#     return np.sum(imfs, axis=0) + residue
# def plot_original_vs_reconstructed(time_series, reconstructed_signal):
#     time_points = np.arange(0, len(time_series))

#     plt.figure(figsize=(12, 6))

#     plt.plot(time_points, time_series, label="Original Signal", alpha=0.7)
#     plt.plot(time_points, reconstructed_signal, label="Reconstructed Signal", linestyle="--")

#     plt.legend()
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.title("Original vs Reconstructed Signal")
#     plt.tight_layout()
#     plt.show()
# def reconstruct_signal_filtered(imfs, residue, num_imfs_to_remove=1):
#     imfs_filtered = imfs[num_imfs_to_remove:, :]
#     return np.sum(imfs_filtered, axis=0) + residue
# reconstructed_signal_filtered = reconstruct_signal_filtered(imfs, residue, num_imfs_to_remove=1)
# plot_original_vs_reconstructed(time_series, reconstructed_signal_filtered)


# In[ ]:





# In[ ]:





# # 建立测试集

# In[5]:


attributes=['duration', 'temp', 'prec',
       'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin',
       'water_regime_AWD','water_regime_continuously flooding', 'water_regime_midseason drainage',
       'water_regime_rainfed']


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(CH4[attributes], CH4["CH4"], test_size=0.2,random_state=0)


# # Standardization

# In[7]:


from sklearn.preprocessing import StandardScaler

# 初始化标准化处理器
scaler_CH4 = StandardScaler()
num_attr = ['duration', 'temp', 'prec', 'density', 'clay', 'totN', 'TOC', 'C/N', 'pH', 'Norg', 'Nin']
# 对CH4数据集的前10列进行标准化处理
x_train[num_attr] = scaler_CH4.fit_transform(x_train[num_attr])
joblib.dump(scaler_CH4, 'scaler_CH4.pkl')


# In[8]:


x_test[num_attr] = scaler_CH4.transform(x_test[num_attr])


# # Hyperparameter optimization
# Bayesian, grid and random search

# In[16]:


#模型指标
def evaluate_model(model,x,y):
    pred=model.predict(x)
    rmse=np.sqrt(mean_squared_error(y,pred))
    r2=metrics.r2_score(y,pred)
    return rmse,r2


# In[29]:


# grid search


# In[101]:


start =time.time()
param_grid=[
    {'n_estimators':[50,100,110,120,130,140,150,200],
#      'max_features':['auto','sqrt','log2'],
     'max_features':[3,6,9,12],
     'max_depth':[5,10,15,20,25]}
]
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True,verbose=1,n_jobs=-1,refit=True)
#refit=True，一旦通过交叉验证找到了最佳估算器，将在整个训练集上重新训练，提供更多的数据很可能提升性能
grid_search.fit(x_train,y_train)
end = time.time()
print('Running time: %s Seconds'%(end-start))


# In[102]:


grid_search.best_params_


# In[103]:


rf= grid_search.best_estimator_
rf.fit(x_train,y_train)


# In[104]:


print(evaluate_model(rf,x_train,y_train))


# In[105]:


print(evaluate_model(rf,x_test,y_test))


# In[106]:


# 训练集交叉验证
cv = cross_val_score(rf,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[36]:


# random search


# In[49]:


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


# In[50]:


random_search.best_params_


# In[51]:


rf= random_search.best_estimator_
rf.fit(x_train,y_train)
print(evaluate_model(rf,x_train,y_train))
print(evaluate_model(rf,x_test,y_test))
# 训练集交叉验证
cv = cross_val_score(rf,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[40]:


# Bayesian Search 


# In[52]:


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


# In[53]:


bayes_search.best_params_


# In[54]:


rf= bayes_search.best_estimator_
rf.fit(x_train,y_train)
print(evaluate_model(rf,x_train,y_train))
print(evaluate_model(rf,x_test,y_test))
# 训练集交叉验证
cv = cross_val_score(rf,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[ ]:





# In[ ]:





# # XGB

# In[55]:


start =time.time()
x=XGBRegressor(random_state=42)
xgbr_param_grid=[
    {'n_estimators':[50,100,200,300,400,500],
     'learning_rate':[0.1,0.2,0.3,0.4],
     'max_depth':[2,4,6,8,10]}]
xgbr_grid_search=GridSearchCV(x,xgbr_param_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True,verbose=1,n_jobs=-1,refit=True)
#记得refit=True
xgbr_grid_search.fit(x_train,y_train)


# In[56]:


xgbr_grid_search.best_params_


# In[57]:


xgbr=xgbr_grid_search.best_estimator_
xgbr.fit(x_train,y_train)


# In[58]:


print(evaluate_model(xgbr,x_test,y_test))
print(evaluate_model(xgbr,x_train,y_train))
# 训练集交叉验证
cv = cross_val_score(xgbr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[48]:


# random search


# In[59]:


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


# In[60]:


print(xgbr_random_search.best_params_)


# In[61]:


xgbr=xgbr_random_search.best_estimator_
xgbr.fit(x_train,y_train)


# In[62]:


print(evaluate_model(xgbr,x_test,y_test))
print(evaluate_model(xgbr,x_train,y_train))
# 训练集交叉验证
cv = cross_val_score(xgbr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[53]:


## Bayesian Search 


# In[63]:


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


# In[64]:


xgbr_bayes_search.best_params_


# In[65]:


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





# In[ ]:





# In[ ]:





# # SVR

# In[20]:


from sklearn.svm import SVR


# In[67]:


start = time.time()

s = SVR(kernel="rbf")
param_grid = [
    {
        'C': [10, 100, 1000, 1500, 2000,10000],
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


# In[68]:


grid_search.best_params_


# In[69]:


svr = grid_search.best_estimator_
svr.fit(x_train,y_train)
print(evaluate_model(svr,x_test,y_test))
print(evaluate_model(svr,x_train,y_train))
cv = cross_val_score(svr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[61]:


# Random Search 


# In[70]:


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
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

random_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[71]:


random_search.best_params_


# In[72]:


svr = random_search.best_estimator_
svr.fit(x_train,y_train)
print(evaluate_model(svr,x_test,y_test))
print(evaluate_model(svr,x_train,y_train))
cv = cross_val_score(svr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[65]:


# Bayesian Search 


# In[66]:


from skopt.space import Real


# In[73]:


start = time.time()

s = SVR(kernel="rbf")
param_dist = {
    'C': Real(10, 10000),  # 设置C为连续空间
    'gamma': Real(0.1, 0.5),  # 设置gamma为连续空间
#     'epsilon': Real(0.1, 0.5),  # 设置epsilon为连续空间
#     'kernel': ['rbf', 'sigmoid', 'poly']
}

bayes_search = BayesSearchCV(
    s,
    search_spaces=param_dist,
    n_iter=50,  # 设置搜索次数
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    verbose=1,
    n_jobs=-1,
    refit=True
)

bayes_search.fit(x_train, y_train)

end = time.time()
print('Running time: %s Seconds' % (end - start))


# In[74]:


bayes_search.best_params_


# In[75]:


svr = bayes_search.best_estimator_
svr.fit(x_train,y_train)
print(evaluate_model(svr,x_test,y_test))
print(evaluate_model(svr,x_train,y_train))
cv = cross_val_score(svr,x_train,y_train,cv=10)
print(cv.mean())
print(cv.std())


# In[ ]:





# In[ ]:





# In[ ]:





# # MLP

# In[19]:


from sklearn.neural_network import MLPRegressor


# In[29]:


mlp=MLPRegressor(early_stopping=True, hidden_layer_sizes=(1000),
             learning_rate_init=0.002, max_iter=500, random_state=42,
             verbose=True)


# In[30]:


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


# In[31]:


grid_search.best_params_


# In[32]:


mlp=grid_search.best_estimator_
mlp.fit(x_train,y_train)

cv = cross_val_score(mlp,x_train,y_train,cv=10)


# In[33]:


print(evaluate_model(mlp,x_test,y_test))
print(evaluate_model(mlp,x_train,y_train))
print(cv.mean())
print(cv.std())


# In[76]:


# Random Search 


# In[34]:


start = time.time()

mlp = MLPRegressor(random_state=42)
param_dist = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'learning_rate': ['constant', 'adaptive'],
    'hidden_layer_sizes':[(100),(150),(200),(100,50),(200,100,50)],
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


# In[35]:


print(random_search.best_params_)


# In[36]:


mlp=random_search.best_estimator_
mlp.fit(x_train,y_train)
cv = cross_val_score(mlp,x_train,y_train,cv=10)


# In[37]:


print(evaluate_model(mlp,x_test,y_test))
print(evaluate_model(mlp,x_train,y_train))

print(cv.mean())
print(cv.std())


# In[81]:


# Bayesian Search 


# In[38]:


from skopt.space import Categorical, Integer, Real


# In[39]:


start = time.time()


param_space = {
    'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
    'learning_rate_init': Real(0.0001, 0.01, prior='log-uniform'),  # 初始学习率
    'learning_rate': Categorical(['constant', 'adaptive']),  # 学习率更新策略
    'hidden_layer_sizes': Integer(50, 300),  # 隐藏层神经元数量

}

bayes_search = BayesSearchCV(
    mlp,
    search_spaces=param_space,
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


# In[40]:


print(bayes_search.best_params_)


# In[41]:


mlp=bayes_search.best_estimator_
mlp.fit(x_train,y_train)

cv = cross_val_score(mlp,x_train,y_train,cv=10)


# In[42]:


print(evaluate_model(mlp,x_test,y_test))
print(evaluate_model(mlp,x_train,y_train))
print(cv.mean())
print(cv.std())


# In[80]:


x_train


# # 保存模型

# In[21]:


rf_CH4 = RandomForestRegressor(max_depth=25, max_features=9, n_estimators=200, n_jobs=1)
xgbr_CH4 = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=400, n_jobs=1)
svr_CH4 = SVR(C=1000, gamma=0.1)
mlp_CH4 = MLPRegressor(activation='tanh', hidden_layer_sizes=(200, ), learning_rate='constant', learning_rate_init=0.01)


# In[22]:


rf_CH4.fit(x_train,y_train)
xgbr_CH4.fit(x_train,y_train)
svr_CH4.fit(x_train,y_train)
mlp_CH4.fit(x_train,y_train)


# In[23]:


joblib.dump(rf_CH4,'rf_CH4.pkl')
joblib.dump(xgbr_CH4,'xgbr_CH4.pkl')
joblib.dump(svr_CH4,'svr_CH4.pkl')
joblib.dump(mlp_CH4,'mlp_CH4.pkl')


# In[1]:


import joblib


# In[ ]:





# In[24]:


rf_CH4


# In[ ]:




