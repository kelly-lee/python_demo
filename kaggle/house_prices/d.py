import numpy as np  # linear algebra
import pandas as pd  #
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import os

from tools import Preprocessing
from tools import ModelTools

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_id = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
print(train.shape)
# train = train[train.GrLivArea < 4500]
print(train.shape)
# train = train[train.LotArea < 100000]
# print(train.shape)
train.reset_index(drop=True, inplace=True)

train["SalePrice"] = np.log1p(train["SalePrice"])
ytrain = train['SalePrice'].reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
#######################################################数据导入和特征提取-【结束】################################################################################

##############################################################特征处理-【开始】###################################################################################
# 对于列名为'MSSubClass'、'YrSold'、'MoSold'的特征列，将列中的数据类型转化为string格式。
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

###############################填充空值##########################
Preprocessing.fillna_mode(features,
                          ['Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'])
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
Preprocessing.fillna_object(features, 'None')
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
Preprocessing.fillna_numeric(features, 0)

######################数字型数据列偏度校正#######################
Preprocessing.boxcox(features)

######################特征删除和融合创建新特征###################

# 融合多个特征，生成新特征。
features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])

# 简化特征。对于某些分布单调（比如100个数据中有99个的数值是0.9，另1个是0.1）的数字型数据列，进行01取值处理。
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

######################特征删除和融合创建新特征-【结束】###################
# 删除一些特征。df.drop（‘列名’, axis=1）代表将‘列名’对应的列标签（们）沿着水平的方向依次删掉。
features = features.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)

####################特征投影、特征矩阵拆解和截取#################

final_features = pd.get_dummies(features).reset_index(drop=True)

mode_info = Preprocessing.info(final_features).sort_values(by=['mode_p'], ascending=False)
drop_col = mode_info.loc[mode_info['mode_p'] > 0.9994].index
final_features = final_features.drop(drop_col, axis=1)

Xtrain = final_features.iloc[:len(ytrain), :]
Xtest = final_features.iloc[len(ytrain):, :]

# 523,1298
outliers = [1324, 1298, 968, 632, 523, 495, 462, 88, 30]  # 88  495 968
# outliers = [30, 88, 462, 631, 1322]
Xtrain = Xtrain.drop(Xtrain.index[outliers])
ytrain = ytrain.drop(ytrain.index[outliers])

##############################################################机器学习-【开始】###################################################################################
print('特征处理已经完成。开始对训练数据进行机器学习', datetime.now())

# 设置k折交叉验证的参数。
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


# 定义均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# 创建模型评分函数，根据不同模型的表现打分
# cv表示Cross-validation,交叉验证的意思。
def cv_rmse(model, X=Xtrain, y=ytrain):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)


#############个体机器学习模型的创建（即模型声明和参数设置）-【开始】############

# 0.1066
# 0.0064 把预测后的误差大的样本删除
# 0.0030 boxcox
# 0.0004 encoder 转 onehot
# 0.0001 新特征
# 0.0000 合理的填充值
# 0.0000 删除多余特征，删错有价值的列，反而会降低模型评分

lasso, ridge, elasticnet = ModelTools.linear_cv_fit(Xtrain, ytrain)
# svr, gbr, xgboost, lightgbm = ModelTools.tree_cv_fit(Xtrain, ytrain)
#
# stack = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, svr, gbr, xgboost, lightgbm),
#                             meta_regressor=xgboost,
#                             use_features_in_secondary=True)
# stack = stack.fit(np.array(Xtrain), np.array(ytrain))

ytrain_pred = lasso.predict(Xtrain)

diff = ytrain_pred - ytrain
diff = abs(diff) / ytrain
sd = diff.sort_values(ascending=False)

# y_predict = (
#         (0.2 * lasso.predict(Xtest)) +
#         (0.15 * ridge.predict(Xtest)) +
#         (0.2 * elasticnet.predict(Xtest)) +
#         (0.15 * svr.predict(Xtest)) +
#         (0.1 * gbr.predict(Xtest)) +
#         (0.1 * xgboost.predict(Xtest)) +
#         (0.1 * lightgbm.predict(Xtest))
# )
y_predict =  elasticnet.predict(Xtest)

y_predict = np.expm1(y_predict)
result = pd.DataFrame({'Id': test_id, 'SalePrice': y_predict})
result.to_csv("submission_12.csv", index=False)
