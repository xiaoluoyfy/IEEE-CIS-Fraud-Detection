# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:27:37 2019

@author: Thinkpad
"""

import os
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import roc_auc_score

train= pd.read_csv('train_transaction.csv', index_col='TransactionID')
test = pd.read_csv('test_transaction.csv', index_col='TransactionID')

sample_submission = pd.read_csv('E:/kaggle_IEEE/data/sample_submission.csv', index_col='TransactionID')



# New groupby
train['days_passed'] = train['TransactionDT'] // 86400
train['days_created'] = train['days_passed'] - train['D1']
train['uid4'] = train['card1'].astype(str) + '_' + train['ProductCD'].astype(str) +'_' + train['P_emaildomain'].astype(str) + '_' + \
                train['addr1'].astype(str) + '_' + train['days_created'].astype(str)
train.loc[train['addr1'].isnull(), 'uid4'] = np.nan



# New groupby
test['days_passed'] = test['TransactionDT'] // 86400
test['days_created'] = test['days_passed'] - test['D1']
test['uid4'] = test['card1'].astype(str) + '_' + test['ProductCD'].astype(str) +'_' + test['P_emaildomain'].astype(str) + '_' + \
                test['addr1'].astype(str) + '_' + test['days_created'].astype(str)
test.loc[test['addr1'].isnull(), 'uid4'] = np.nan

train['daypassed_minus_D15']=train['days_passed']-train['D15']
test['daypassed_minus_D15']=test['days_passed']-test['D15']

train['daypassed_plus_D15']=train['days_passed']+train['D15']
test['daypassed_plus_D15']=test['days_passed']+test['D15']

train['daypassed_minus_D2']=train['days_passed']-train['D2']
test['daypassed_minus_D2']=test['days_passed']-test['D2']

train[['daypassed_minus_D15','daypassed_plus_D15','daypassed_minus_D2' ]].to_csv('daypassed_features_train_df_added.csv',index=None)
test[['daypassed_minus_D15','daypassed_plus_D15','daypassed_minus_D2' ]].to_csv('daypassed_features_test_df_added.csv',index=None)


