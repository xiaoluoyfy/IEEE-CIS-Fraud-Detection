# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 14:23:55 2019

@author: Thinkpad
"""

#------------------------------step 01--------------------------------------------
import pandas as pd

res1=pd.read_csv('lgb_kfold_0.95365-9605.csv')  #-----9605--------
res2=pd.read_csv('lgb_kfold-9614.csv')      #-----9414--------

res=pd.DataFrame()
res['TransactionID']=res1['TransactionID'].values

res['isFraud']=0.35*res1['isFraud']+0.65*res2['isFraud']
res.to_csv('lgb_890features_blend_0.65.csv',index=None)  #---9646
res.head()


#------------------------------step 02--------------------------------------------



import pandas as pd

   
res1=pd.read_csv('lgb_890features_blend_0.65.csv')   #----9646
nn= pd.read_csv('CV9518_NN_LB9556.csv',index_col=0) #LB 9556
nn.columns=['isFraud']

oof_nn=nn[:590540]
pred_nn= nn[590540:]

res=pd.DataFrame()
res['TransactionID']=res1['TransactionID'].values
res['isFraud']=0.95*res1['isFraud'].values+0.05*pred_nn['isFraud'].values
res.to_csv('lgb_0930_0.95_v0.csv',index=None)                    #----9663------
res.head()



#------------------------------step 03--------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#nn= pd.read_csv('CV9518_NN_LB9556.csv',index_col=0)
nn=pd.read_csv('NNCV9548_oofandTest',index_col=0)
nn.columns=['isFraud']

lgb= pd.read_csv('lgb_cv9562_692features_oofAndTest.csv',index_col=0)
lgb.columns=['isFraud']

ctb=pd.read_csv('CatBoost_cv9582_692features_oofAndTest.csv',index_col=0)
ctb.columns=['isFraud']

oof_nn=nn[:590540]
pred_nn= nn[590540:]

oof_lgb=lgb[:590540]
pred_lgb= lgb[590540:]

oof_ctb=ctb[:590540]
pred_ctb= ctb[590540:]


train= pd.read_csv('train_transaction.csv', index_col='TransactionID')
sample_submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')

y_train=train['isFraud'].copy()


print(roc_auc_score(y_train.values, 0.33*oof_ctb.values+0.62*oof_lgb.values+0.05*oof_nn.values))
temp=pd.DataFrame()
temp['TransactionID']=sample_submission.index
temp['isFraud']=0.33*pred_ctb.values+0.62*pred_lgb.values+0.05*pred_nn.values
temp.to_csv('pred_692_features_blend.csv')



#------------------------------step 04--------------------------------------------
import pandas as pd
res1=pd.read_csv('lgb_0930_0.95_v0.csv')
res2=pd.read_csv('pred_692_features_blend.csv')
res=pd.DataFrame()
res['TransactionID']=res1['TransactionID'].values
res['isFraud']=0.65*res1['isFraud']+0.35*res2['isFraud']
res.to_csv('lgb_0930_0.65_v1.csv',index=None) 

















