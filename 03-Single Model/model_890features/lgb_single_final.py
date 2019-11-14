# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


import warnings
warnings.filterwarnings("ignore")
gc.enable()

import seaborn as sns
import matplotlib.pyplot as plt
from time import time, strftime, localtime
from contextlib import contextmanager

@contextmanager
def timer(name): 
    start = time()
    yield
    print(f'[{name}] done in {time() - start:.2f} s')

def show_cur_time():
    print(strftime('%Y-%m-%d %H:%M:%S',localtime(time())))
 
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

IS_DEBUG = False 
IS_REDUCE = False
IS_ADV_VAL = True
IS_KS_VAL = False
proj_path = "D://kaggle//ieee-fraud-detection"

with timer('load data'):
    show_cur_time()
    print("load data ...")
    if IS_DEBUG:
        n_rows = 10000
    else:
        n_rows = None
        
    if 'train_by_id.pkl' in os.listdir(os.path.join(proj_path, 'input')) and \
                    'test_by_id.pkl' in os.listdir(os.path.join(proj_path, 'input')) :
        t = time()
        with open(os.path.join(proj_path, "input/train_by_id.pkl"), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(proj_path, "input/test_by_id.pkl"), 'rb') as f:
            test = pickle.load(f)
        sub = pd.read_csv(os.path.join(proj_path, 'input/sample_submission.csv'), nrows=n_rows)
        print("loading data finished in {}s".format(time() - t))
        if IS_DEBUG:
            train = train.iloc[:n_rows]
            test = test.iloc[:n_rows]
    else:
        t = time()
        train_identity= pd.read_csv(os.path.join(proj_path, "input/train_identity.csv"), nrows=n_rows, index_col='TransactionID')
        train_transaction= pd.read_csv(os.path.join(proj_path, "input/train_transaction.csv"), nrows=n_rows, index_col='TransactionID')
        test_identity= pd.read_csv(os.path.join(proj_path, "input/test_identity.csv"), nrows=n_rows, index_col='TransactionID')
        test_transaction = pd.read_csv(os.path.join(proj_path, "input/test_transaction.csv"), nrows=n_rows, index_col='TransactionID')
        sub = pd.read_csv(os.path.join(proj_path, 'input/sample_submission.csv'), nrows=n_rows, index_col='TransactionID')
        
        
        def id_split(dataframe):
            dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
            dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]
        
            dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
            dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]
        
            dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
            dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]
        
            dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
            dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]
        
            dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
            dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]
        
            dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
            dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
            dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
            dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
            dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
            dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
        
            dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
            dataframe['had_id'] = 1
            gc.collect()
            
            return dataframe
        
        train_identity = id_split(train_identity)
        test_identity = id_split(test_identity)
        

        train = train_transaction.merge(train_identity, how='left', on='TransactionID')
        test = test_transaction.merge(test_identity, how='left', on='TransactionID')
        
        del train_identity,train_transaction,test_identity, test_transaction
        gc.collect()
        
        print('dumping data to pkl format')
        with open(os.path.join(proj_path, "input/train_by_id.pkl"), 'wb') as f:
            pickle.dump(train, f)
        with open(os.path.join(proj_path, "input/test_by_id.pkl"), 'wb') as f:
            pickle.dump(test, f)
    if IS_REDUCE:
        train=reduce_mem_usage(train)
        test=reduce_mem_usage(test)
    print('training set shape:', train.shape)
    print('test set shape:', test.shape)

trn_length = train.shape[0]
test['isFraud'] = -1
data = pd.concat([train, test], axis=0)
del train, test
    
cols_to_drop = ['C3', 'C7', 'D7', 'D9', 'D12', 'D13', 'D14', 'M1', 'M2', 'M3', 'M7', 'M8', 'M9', 
                'V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17', 
                'V18', 'V19', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32',
                'V33', 'V34', 'V35', 'V37', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V46', 'V47', 'V48', 'V49',
                'V50', 'V51', 'V52', 'V57', 'V58', 'V59', 'V60', 'V63', 'V64', 'V65', 'V66', 'V68', 'V69', 'V71',
                'V72', 'V73', 'V74', 'V77', 'V79', 'V80', 'V81', 'V84', 'V85', 'V86', 'V88', 'V89', 'V90', 'V91',
                'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 
                'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 
                'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V128', 'V129', 
                'V132', 'V134', 'V135', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V144', 'V145', 'V146', 
                'V147', 'V148', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 
                'V161', 'V163', 'V164', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 
                'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 
                'V188', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 
                'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V210', 'V211', 'V212', 'V213', 
                'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 
                'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 
                'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 
                'V250', 'V252', 'V253', 'V254', 'V255', 'V256', 'V259', 'V260', 'V261', 'V263', 'V264', 'V265', 
                'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V278', 'V279', 
                'V280', 'V284', 'V286', 'V287', 'V288', 'V289', 'V290', 'V292', 'V293', 'V295', 'V297', 'V298', 
                'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V316', 'V318', 'V319', 'V321', 
                'V322', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 
                'V335', 'V336', 'V337', 'V338', 'V339', 'id_04', 'id_06', 'id_07', 'id_08', 'id_10', 'id_11', 
                'id_12', 'id_15', 'id_16', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 
                'id_27', 'id_28', 'id_29', 'id_32', 'id_34', 'id_35', 'id_36', 'id_37']

print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

show_cur_time()
data = data.drop(cols_to_drop, axis=1)

data['P_isproton']=(data['P_emaildomain']=='protonmail.com')
data['R_isproton']=(data['R_emaildomain']=='protonmail.com')

a = np.zeros(data.shape[0])
data["lastest_browser"] = a
def setbrowser(df):
    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
    return df
data=setbrowser(data)
del data['id_31']
del data['version_id_31']

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 
          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 
          'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum',
          'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 
          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 
          'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 
          'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 
          'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 
          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 
          'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 
          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 
          'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 
          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 
          'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 
          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 
          'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 
          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 
          'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 
          'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']
for c in ['P_emaildomain', 'R_emaildomain']:
    data[c + '_bin'] = data[c].map(emails)
    data[c + '_suffix'] = data[c].map(lambda x: str(x).split('.')[-1])
    data[c + '_suffix'] = data[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    
data['P_R_email_match'] = (data['P_emaildomain'] == data['R_emaildomain'])

data['TransactionAmt_decimal'] = ((data['TransactionAmt'] - data['TransactionAmt'].astype(int)) * 1000).astype(int)
data['TransactionAmt_Log'] = np.log1p(data['TransactionAmt'])
data['amt_to_prod_median_ratio'] = data['TransactionAmt']/data['ProductCD'].map(data[['ProductCD','TransactionAmt']].groupby(['ProductCD'])['TransactionAmt'].median())

data['Transaction_day_of_week'] = np.floor((data['TransactionDT'] / (3600 * 24) - 1) % 7)
data['Transaction_hour_of_day'] = np.floor(data['TransactionDT'] / 3600) % 24
data['Date'] = (data['TransactionDT'] / (3600 * 24) // 1)

# low frequency filtering on original features
long_tail_cols = ['card1', 'addr1', ] + ['addr2',]
filter_thres = 1 
print("perform low freq filtering on long-tailed categorical features:{}...".format(long_tail_cols))
for col in tqdm(long_tail_cols): 
    valid_col = data[col].value_counts()
    valid_col = valid_col[valid_col>filter_thres]
    valid_col = list(valid_col.index)

    train = data[:trn_length]
    test = data[trn_length:]

    train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
    test[col]  = np.where(test[col].isin(train[col]), test[col], np.nan)

    train[col] = np.where(train[col].isin(valid_col), train[col], np.nan)
    test[col]  = np.where(test[col].isin(valid_col), test[col], np.nan)
    data = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()

data['user_lvl_1'] = data['card1'].astype(str) +'_'+ data['card2'].astype(str)+\
                    '_'+data['card3'].astype(str)+'_'+data['card4'].astype(str)+'_'+data['card5'].astype(str)+'_'+\
                    data['card6'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)+\
                    '_'+data['P_emaildomain'].astype(str)

data['user_lvl_2'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)+\
                    '_'+data['card3'].astype(str)+'_'+data['card4'].astype(str)+'_'+data['card5'].astype(str)+'_'+\
                    data['card6'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

data['user_lvl_3'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)+\
                    '_'+data['card3'].astype(str)+'_'+data['card4'].astype(str)+'_'+data['card5'].astype(str)+'_'+\
                    data['card6'].astype(str)+'_'+data['addr1'].astype(str)

data['user_lvl_4'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)+\
                    '_'+data['card3'].astype(str)+'_'+data['card4'].astype(str)+'_'+data['card5'].astype(str)+'_'+\
                    data['card6'].astype(str)

data['days_created'] = data['Date'] - data['D1']
data['uid1'] = data['card1'].astype(str) + '_' + data['days_created'].astype(str)
data['uid2'] = data['card2'].astype(str)  + '_' + data['days_created'].astype(str)
data['uid3'] = data['addr1'].astype(str) + '_' + data['days_created'].astype(str)   
#data['uid4'] = data['addr2'].astype(str) + '_' + data['days_created'].astype(str)
data['uid5'] = data['addr1'].astype(str) + '_' + data['days_created'].astype(str) + '_' + data['ProductCD'].astype(str)
data['uid6'] = data['P_emaildomain'].astype(str) + '_' + data['days_created'].astype(str)  + '_' + data['ProductCD'].astype(str)
data['uid7'] = data['card1'].astype(str) + '_' + data['days_created'].astype(str) + '_' + data['ProductCD'].astype(str)
data['uid8'] = data['user_lvl_1'].astype(str) + '_' + data['days_created'].astype(str)
data['uid9'] = data['user_lvl_2'].astype(str) + '_' + data['days_created'].astype(str)
data['uid10'] = data['user_lvl_3'].astype(str) + '_' + data['days_created'].astype(str)
data['uid11'] = data['user_lvl_4'].astype(str) + '_' + data['days_created'].astype(str)

uid_levels = ['uid1', 'uid2', 'uid3',] + ['uid5', 'uid6', 'uid7'] + ['uid8', 'uid9', 'uid10', 'uid11'] 
del data['days_created']
gc.collect()

freq_encode_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'P_emaildomain', 'R_emaildomain', ]
for col in tqdm(freq_encode_cols):
    if col in data.columns:
        data[col+'_count_full'] = data[col].map(data[col].value_counts(dropna=False))

for col in ['D1','D2','D3','D4','D5','D6','D7','D8','D10','D11','D12','D13','D14','D15']:
    if col in data.columns:
        data[col+'_rank']=data.groupby(['Date'])[col].rank(ascending=False) / data.groupby(['Date'])[col].transform('count')
        data[col]=(data[col]-data.groupby(['Date', 'card1'])[col].transform('min')) / (data.groupby(['Date', 'card1'])[col].transform('max')-data.groupby(['Date', 'card1'])[col].transform('min'))


user_level_cols = ['user_lvl_1', 'user_lvl_2', 'user_lvl_3', 'user_lvl_4',]
for col in tqdm(user_level_cols + uid_levels ):
    if col in data.columns:
        data[col+'_count_full'] = data[col].map(data[col].value_counts(dropna=False))

for level in tqdm(user_level_cols + uid_levels):
    data[level + '_amt_sum'] = data[level].map(data[[level,'TransactionAmt']].groupby([level])['TransactionAmt'].sum())
    data[level + '_amt_sum_ratio'] = data['TransactionAmt'] /  data[level + '_amt_sum']
    
for level in tqdm(user_level_cols + uid_levels):
    data[level + '_amt_mean'] = data[level].map(data[[level,'TransactionAmt']].groupby([level])['TransactionAmt'].mean())
    data[level + '_amt_mean_ratio'] = data['TransactionAmt'] /  data[level + '_amt_mean']
    data[level + '_amt_mean_diff'] = data['TransactionAmt'] -  data[level + '_amt_mean']

for level in tqdm(user_level_cols + uid_levels):
    data[level + '_amt_median'] = data[level].map(data[[level,'TransactionAmt']].groupby([level])['TransactionAmt'].median())
    data[level + '_amt_median_ratio'] = data['TransactionAmt'] /  data[level + '_amt_median']
    data[level + '_amt_median_diff'] = data['TransactionAmt'] -  data[level + '_amt_median']
    
for level in tqdm(user_level_cols + uid_levels):
    data[level + '_amt_max'] = data[level].map(data[[level,'TransactionAmt']].groupby([level])['TransactionAmt'].max())
    data[level + '_amt_min'] = data[level].map(data[[level,'TransactionAmt']].groupby([level])['TransactionAmt'].min())
    data[level + '_amt_max_ratio'] = data['TransactionAmt'] /  data[level + '_amt_max']
    data[level + '_amt_max_diff'] = data['TransactionAmt'] -  data[level + '_amt_max']
    data[level + '_amt_min_ratio'] = data['TransactionAmt'] /  data[level + '_amt_min']
    data[level + '_amt_min_diff'] = data['TransactionAmt'] -  data[level + '_amt_min']


for level in tqdm(user_level_cols + uid_levels):    
    data[level + '_timedelta_to_last'] = data[[level,'TransactionDT']].groupby([level])['TransactionDT'].shift(1)
    data[level + '_timedelta_to_last'] = data['TransactionDT'] - data[level + '_timedelta_to_last']
    data[level + '_timedelta_to_next'] = data[[level,'TransactionDT']].groupby([level])['TransactionDT'].shift(-1)
    data[level + '_timedelta_to_next'] = data['TransactionDT'] - data[level + '_timedelta_to_next']
    if level not in ['uid1', 'uid2', 'uid3', 'uid5', 'uid6']:
        data[level + '_timedelta_to_next_mean'] = data[level].map(data[[level, level + '_timedelta_to_next']].groupby([level])[level + '_timedelta_to_next'].mean())
        data[level + '_timedelta_to_next_std'] = data[level].map(data[[level, level + '_timedelta_to_next']].groupby([level])[level + '_timedelta_to_next'].std())
        data[level + '_timedelta_to_last_mean'] = data[level].map(data[[level, level + '_timedelta_to_last']].groupby([level])[level + '_timedelta_to_last'].mean())
        data[level + '_timedelta_to_last_std'] = data[level].map(data[[level, level + '_timedelta_to_last']].groupby([level])[level + '_timedelta_to_last'].std())

for level in tqdm(user_level_cols + uid_levels ):
    data[level + '_dist1_std'] = data[level].map(data[[level,'dist1']].groupby([level])['dist1'].std())
    data[level + '_dist1_mean'] = data[level].map(data[[level,'dist1']].groupby([level])['dist1'].mean())
    data[level + '_dist1_max'] = data[level].map(data[[level,'dist1']].groupby([level])['dist1'].max())
    data[level + '_dist1_min'] = data[level].map(data[[level,'dist1']].groupby([level])['dist1'].min())

for level in tqdm(user_level_cols + uid_levels ):
    data[level + '_dist2_std'] = data[level].map(data[[level,'dist2']].groupby([level])['dist2'].std())
    data[level + '_dist2_mean'] = data[level].map(data[[level,'dist2']].groupby([level])['dist2'].mean())

for level in tqdm(user_level_cols):
    data[level + '_date_count'] = data[level].map(data[[level,'Date']].groupby([level])['Date'].nunique())

for level in tqdm(user_level_cols + uid_levels ):
    data[level + '_hour_cnt'] = data[level].map(data[[level,'Transaction_hour_of_day']].groupby([level])['Transaction_hour_of_day'].nunique())

for level in tqdm(user_level_cols + uid_levels ):
    data['hour_' + level + '_cnt'] = data['Transaction_hour_of_day'].map(data[[level,'Transaction_hour_of_day']].groupby(['Transaction_hour_of_day'])[level].nunique())

for card in tqdm(['card1','card2','addr1',]):
    for level in (user_level_cols + uid_levels):   
        data[card+'_'+level+'_cnt'] = data[card].map(data[[card,level]].groupby([card])[level].nunique())


for card in tqdm(['card1','card2','addr1',]):
    data[card+'_dist1_mean'] = data[card].map(data[[card,'dist1']].groupby([card])['dist1'].mean())
    data[card+'_dist2_mean'] = data[card].map(data[[card,'dist2']].groupby([card])['dist2'].mean())
    data[card+'_dist1_std'] = data[card].map(data[[card,'dist1']].groupby([card])['dist1'].std())
    data[card+'_dist2_std'] = data[card].map(data[[card,'dist2']].groupby([card])['dist2'].std())
    
for card in tqdm(['card1','card2','addr1',]):
    data[card+'_amt_mean'] = data[card].map(data[[card,'TransactionAmt']].groupby([card])['TransactionAmt'].mean())
    data[card+'_amt_std'] = data[card].map(data[[card,'TransactionAmt']].groupby([card])['TransactionAmt'].std())

cross_feature = ['card1__card2','card1__addr1','card1__addr2','card2__addr1','card2__addr2','addr1__addr2'] + \
                ['card2__P_emaildomain',] + ['addr1__id_20', 'addr1__P_emaildomain']

for card in tqdm(cross_feature):
    card1=card.split('__')[0]
    card2=card.split('__')[1]
    data[card]=data[card1].astype(str)+'_'+data[card2].astype(str)
    data[card+'_count_full'] = data[card].map(data[card].value_counts(dropna=False)) 
    data[card+'_amt_mean'] = data[card].map(data[[card,'TransactionAmt']].groupby([card])['TransactionAmt'].mean())
    data[card+'_amt_std'] = data[card].map(data[[card,'TransactionAmt']].groupby([card])['TransactionAmt'].std())

C_cols = ['C1', 'C2', 'C9', 'C12', 'C13',]
id_con_cols = ['id_02', 'id_05']
id_cat_cols = ['id_19', 'id_20']
for col in tqdm(C_cols + id_con_cols + id_cat_cols):
    if col in data.columns:
        data[col+'_count_full'] = data[col].map(data[col].value_counts(dropna=False))    

for group in tqdm(['card1','card2','addr1',] + user_level_cols + uid_levels):
    for target in (C_cols + id_con_cols):
        data[group+'_'+target+'_mean'] = data[group].map(data[[group, target]].groupby([group])[target].mean())
        data[group+'_'+target+'_std'] = data[group].map(data[[group, target]].groupby([group])[target].std())

for col in uid_levels:
    del data[col]
train = data[:trn_length]
test = data[trn_length:]
del data
gc.collect()


show_cur_time()

if IS_ADV_VAL:
    train_samples = train.sample(10000, random_state=1994)
    test_samples = test.sample(10000, random_state=1994)
    selected_cols = [col for col in train.columns if col != 'isFraud']
    train_samples = train_samples[selected_cols]
     
    train_samples['origin'] = 0
    test_samples['origin'] = 1

    train_samples = train_samples.replace([np.inf, -np.inf], np.nan)
    test_samples = test_samples.replace([np.inf, -np.inf], np.nan)
    train_samples = train_samples.fillna(-999)
    test_samples = test_samples.fillna(-999)

    print('Label Encoding...')
    for f in train_samples.columns:
        if f != 'isFraud':
            if train_samples[f].dtype=='object' or train_samples[f].dtype=='object':
                lbl = LabelEncoder()
                lbl.fit(list(train_samples[f].values)+ list(test_samples[f].values))
                train_samples[f] = lbl.transform(list(train_samples[f].values))
                test_samples[f] = lbl.transform(list(test_samples[f].values))
    
    combi = train_samples.append(test_samples)
    y_samples = combi['origin']
    if 'isFraud' in combi.columns:
        print("aha! target found in features!Remove!")
        combi.drop('isFraud',axis=1,inplace=True)
    combi.drop('origin',axis=1,inplace=True)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)
    all_scores = []
    temp = -1
    for i in combi.columns:
        temp +=1
        score = cross_val_score(model,pd.DataFrame(combi[i]),y_samples,cv=2,scoring='roc_auc')
        all_scores.append(np.mean(score))    
    scores_df = pd.DataFrame({'feature':combi.columns, 
                              'score': all_scores})
    scores_df = scores_df.sort_values(by = 'score', ascending = False)  


weights = 0.998**(train['Date'].max()-train['Date'])
del train['Date'],test['Date'],train['TransactionDT'],test['TransactionDT']

y_train = train['isFraud'].copy()
oof_pred = train['isFraud'].copy()
X_train = train.drop('isFraud', axis=1)
X_test = test.drop('isFraud', axis=1)
del train, test

          
print("label encoding for categorical features...")
        
for col in tqdm(X_train.columns):
    if X_train[col].dtype =='object' or X_test[col].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(pd.concat([X_train[col].astype(str),X_test[col].astype(str)]))
        X_train[col] = lbl.transform(X_train[col].astype(str))
        X_test[col] = lbl.transform(X_test[col].astype(str))  

        
n_fold = 5
folds = KFold(n_splits=n_fold)

show_cur_time()
sub['isFraud'] = 0

lgb_params = {
            'objective':'binary',
            'boosting_type':'gbdt',
            'metric':'auc',
            'n_jobs':-1,
            'learning_rate':0.01,
            'num_leaves': 2**8,
            'max_depth': -1,
            'tree_learner':'serial',
            'colsample_bytree': 0.7,
            'subsample_freq':1,
            'subsample':0.8,
            'max_bin':255,
            'verbose':-1,
            'seed': 1994,
                } 


num_round=10000
score=0
feature_importances = pd.DataFrame()
feature_importances['feature'] = X_train.columns
for fold, (train_index, valid_index) in enumerate(folds.split(X_train)):
    print(fold, valid_index[0], valid_index[-1])
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    trn_weights = weights.iloc[train_index]
    trn_data = lgb.Dataset(X_train_, label=y_train_, weight=trn_weights)
    val_data = lgb.Dataset(X_valid, label=y_valid) 
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=100, early_stopping_rounds = 100)
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    del X_train_,y_train_
    pred=clf.predict(X_test, num_iteration=clf.best_iteration)
    val=clf.predict(X_valid, num_iteration=clf.best_iteration)
    oof_pred.iloc[valid_index] = val
    del X_valid
    print('AUC: {}'.format(roc_auc_score(y_valid, val)))
    score += roc_auc_score(y_valid, val) / n_fold
    del val, y_valid
    sub['isFraud'] = sub['isFraud'] + pred / n_fold
    del pred
    gc.collect()
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('Mean AUC by folds:', score)
print('Overall AUC by oof:', roc_auc_score(y_train, oof_pred))
print("oof statistics:{} ± {}".format(oof_pred.mean(), oof_pred.std()))
feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));

sub.to_csv(os.path.join(proj_path, 'submit/lgb_kfold_'+str(score)[:7]+'.csv'), index=False)
oof_pred.to_csv(os.path.join(proj_path, 'submit/lgb_kfold_'+str(score)[:7]+'_oof.csv'), index=False)
