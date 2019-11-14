## Requirements

python 3.7.3

numpy 1.16.2

pandas 0.24.2

sklearn 0.20.3

keras 2.2.4

tensorflow 1.13.1

xgboost 0.82

lightgbm 2.2.3



## How to reproduce
### 01-Input Model_Data

* Put unzipped data/model data in `01-Input Model_Data`
* [all the model data can be found in google drive](https://drive.google.com/drive/folders/13xt6QpbxvTVwZl7h-Za1-EREcy5i7eln?usp=sharing)
* Generate a simple solution that is good enough for 3rd place (~0.943642 on private LB)

### 02-Feature Enginnering
* 890 features<br>
`cd /02-Feature_Enginnering/890features/`<br>
`python lgb_single_final.py`----Also output the result

* 692 features<br>
`cd /02-Feature Enginnering/692features/1.baseline_features_388/`<br>
`python 1.feature engineering.py`<br>
`python 2.feature selection.py`<br>
`python 3.feature engineering.py`<br>
`cd /02-Feature Enginnering/692features/2.uid_magic_features_301/`<br>
`python uid4_eng.py`<br>
`cd /02-Feature Enginnering/692features/3.combine_features_3/`<br>
`python combine_features_3.py`<br>


### 03-Single Model
* 890 features<br>
`cd /03-Single_Model/890features/`<br>
`python lgb_single_final.py`<br>
-----------------CV 95365 LB 9605

* 692 features<br>
`cd /03-Single_Model/692features/`<br>
`python Lgb_CV9562_LB9597.py`<br>
-----------------CV 9562 LB 9597<br>
----------------- tune the parameters the lgb can reach LB 9614<br>
`python Catboost_CV9582_LB9590.py`<br>
-----------------CV 9582 LB 9590<br>
`python NN_CV9518_LB9556.py`<br>
-----------------CV 9518 LB 9556<br>

### Model Blend
* `cd /04-Model Blend/`<br>
  `python model_blend.py`<br>
* **Finally：lgb_0930_0.65_v1** <br> 
* **Public:0.967161  Private:0.943642**<br>

```
step 01:
lgb_890features_blend_0.65=0.65*lgb_kfold_9614+0.35*lgb_kfold_9605
------LB 9646-----
step 02:
lgb_0930_0.95_v0=0.95*lgb_890features_blend_0.65+0.05*CV9518_NN_LB9556
------LB 9663-----
step 03:
lgb_0930_0.65_v1 =0.65*lgb_0930_0.95_v0.csv+0.35*pred_692_features_blend
while:
pred_692_features_blend=0.65*lgb_cv9562_692features_9597+0.35*CatBoost_cv9582_692features_9590
```











  

  

