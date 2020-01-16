import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from evaluation import Ecaluation_category
def encode(data):
    data = pd.get_dummies(data, columns=['poutcome', 'contact'])
    data.drop(columns=['poutcome_unknown','contact_unknown'],inplace=True)
    data['marital'].replace({'single': 0, 'married': 1, 'divorced': 2}, inplace=True)
    data['education'].replace({'unknown': 0, 'primary': 1, 'tertiary': 2, 'secondary': 3}, inplace=True)
    data['default'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['housing'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['loan'].replace({'no': 0, 'yes': 1}, inplace=True)
    # data['contact'].replace({'unknown': 0, 'cellular': 1, 'telephone': 2}, inplace=True)
    # data['poutcome'].replace({'unknown': 0, 'success': 1, 'failure': 2, 'other': 3}, inplace=True)
    data['month'].replace({'jan': 1
                              , 'feb': 2
                              , 'mar': 3
                              , 'apr': 4
                              , 'may': 5
                              , 'jun': 6
                              , 'jul': 7
                              , 'aug': 8
                              , 'sep': 9
                              , 'oct': 10
                              , 'nov': 11
                              , 'dec': 12
                           }, inplace=True)
    data['y'].replace({'no': 0, 'yes': 1}, inplace=True)
    data['job'] = data.job.astype('category').cat.codes
    data['job'] = data['job'].astype('int64')
    return data


bank=pd.read_csv('bank.csv',sep=';')

bank=encode(bank)





X_train,X_test,y_train,y_test= train_test_split(bank.iloc[:,:-1],bank.iloc[:,-1],test_size=0.3,random_state=10)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'f1-macro'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}





print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)







print('Save model...')
# save model to file
# gbm.save_model('lightgbm/model.txt')
print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
print('The roc of prediction is:', roc_auc_score(y_test, y_pred))
print('Dump model to JSON...')
# dump model to json (and save to file)
model_json = gbm.dump_model()
with open('lightgbm/model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)
print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))


