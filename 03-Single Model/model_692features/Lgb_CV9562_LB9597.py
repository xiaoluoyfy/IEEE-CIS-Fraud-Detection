import numpy as np
import pandas as pd
# Data processing, metrics and modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
#from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
# Lgbm
import lightgbm as lgb
# Suppr warning
import warnings
warnings.filterwarnings("ignore")
import itertools
from scipy import interp
# Plots
import seaborn as sns
import matplotlib.pyplot as plt

param_lgb = {'subsample': 0.95,
            'colsample_bytree': 0.9,
            'max_depth': 50,
            'min_child_weight': 0.0029805017044362268,
            'min_child_samples': 10,
            'num_leaves': 381, # use 381
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'n_estimators':3000,
            'learning_rate': 0.03,
            'early_stopping_rounds': 200,
            'first_metric_only': True,
            #'class_weight':'balanced',
               'objective': 'binary',
               'save_binary': True,
               'seed': 1337,
               'feature_fraction_seed': 1337,
               'bagging_seed': 1337,
               'drop_seed': 1337,
               'data_random_seed': 1337,
               'boosting_type': 'gbdt',
               'verbose': 100,
               'is_unbalance': False,
               'boost_from_average': True,
               'metric':'auc'
   }

'''				
param_lgb = {    'min_data_in_leaf': 100, 
                'num_leaves': 300, 
                'learning_rate': 0.008,
                'min_child_weight': 0.03454472573214212,
#                    'bagging_fraction': 0.8, 
               'bagging_fraction': 0.4181193142567742, 
                'feature_fraction': 0.3797454081646243,
#                    'reg_lambda': 0.6485237330340494,
#                    'reg_alpha': 0.3899927210061127,
                'max_depth': -1, 
                'objective': 'binary',
                'seed': SEED,
                'boosting_type': 'gbdt',
                'verbose': 1,
                'metric':'auc',
        }

#------lgb 9614--------		
'''


train_df=pd.read_pickle('E:/IEEE_top3/model_data/train_df_692features_lb9597.pkl')
test_df=pd.read_pickle('E:/IEEE_top3/model_data/test_df_692features_lb9597.pkl')

features=list(test_df.columns)
target = 'isFraud'

nfold = 5
#skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
skf = KFold(n_splits=nfold, shuffle=False, random_state=42)
oof = np.zeros(len(train_df))
mean_fpr = np.linspace(0,1,100)
cms= []
tprs = []
aucs = []
y_real = []
y_proba = []
recalls = []
roc_aucs = []
f1_scores = []
accuracies = []
precisions = []
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()
i = 1
clf = lgb.LGBMClassifier()
clf.set_params(**param_lgb)
for train_idx, valid_idx in skf.split(train_df, train_df.isFraud.values):
   print("\nfold {}".format(i))
   x_train = train_df.iloc[train_idx][features]
   y_train = train_df.iloc[train_idx][target]
   x_val = train_df.iloc[valid_idx][features]
   y_val = train_df.iloc[valid_idx][target]
#         clf = lgb.LGBMClassifier()
#         clf.set_params(**param_lgb)
   clf.fit(x_train, y_train,
		   eval_metric =['auc'],
		   verbose = 100,
		   eval_set = (x_val, y_val) )
   oof[valid_idx] = clf.predict_proba(x_val)[:,1]
   predictions += clf.predict_proba(test_df[features])[:,1]  / nfold
   # Scores
   roc_aucs.append(roc_auc_score(train_df.iloc[valid_idx][target].values, oof[valid_idx]))
   accuracies.append(accuracy_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
   recalls.append(recall_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
   precisions.append(precision_score(train_df.iloc[valid_idx][target].values ,oof[valid_idx].round()))
   f1_scores.append(f1_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
   # Roc curve by folds
   f = plt.figure(1)
   fpr, tpr, t = roc_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])
   tprs.append(interp(mean_fpr, fpr, tpr))
   roc_auc = auc(fpr, tpr)
   aucs.append(roc_auc)
   plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))
   # Precion recall by folds
   g = plt.figure(2)
   precision, recall, _ = precision_recall_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])
   y_real.append(train_df.iloc[valid_idx][target].values)
   y_proba.append(oof[valid_idx])
   plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))
   i= i+1
   # Confusion matrix by folds
   cms.append(confusion_matrix(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
   # Features imp
   fold_importance_df = pd.DataFrame()
   fold_importance_df["Feature"] = features
   fold_importance_df["importance"] = pd.DataFrame(sorted(zip(clf.feature_importances_,x_train.columns)),
												   columns=['Value','Feature'])['Value']
   fold_importance_df["fold"] = nfold + 1
   feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

   # Metrics
print(
   '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
   '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
   '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
   '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
   '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
   )


all_pred = np.concatenate([oof,predictions])
all_pred = pd.DataFrame(all_pred)
all_pred.to_csv('lgb_cv9562_692features_oofAndTest.csv')
sub = pd.read_csv('sample_submission.csv')
sub.isFraud = predictions
sub.to_csv('lgb_cv9562_692features_TestOnly.csv', index=False)