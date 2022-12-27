#%% Importok
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from pandas.api.types import is_numeric_dtype
import os
import joblib

import polars as pl

from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
# print('I see %i GPU devices' % get_gpu_device_count())

#EDA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import time

os.chdir('c:\\ML\\amex')

import pickle

import warnings
warnings.filterwarnings('ignore')

from matplotlib.ticker import MaxNLocator
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model

# Configure notebook display settings to only use 2 decimal places, tables look nicer.
pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

SEED = 42
random.seed(SEED)

def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

# ====================================================
# LGBM amex metric
# ====================================================
def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True

class CFG:
    input_dir = '/content/data/'
    seed = 42
    n_folds = 5
    target = 'target'

#%% POLARS Load test data
print("Adatok betöltése indul!")
start_time = time.time()
train_data = pl.read_parquet('train_data.parquet')
train_lbls = pd.read_csv('train_labels.csv').set_index('customer_ID')
print("Futási idő:", int(time.time()-start_time), "s")


# %% FEATURE engineering + AGG DEF

cat_features = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
    "B_31",
]

cols_to_cat = [ 'D_87']
cat_features +=cols_to_cat


def preprocess_df(df):
    df['S_2'] = pd.to_datetime(df['S_2'])
    df['S_2_max'] = df.groupby('customer_ID').S_2.transform('max')
    df['S_2_diff'] = df.groupby('customer_ID').S_2.transform('diff').dt.days
    df['S_2_day_of_week'] = df['S_2'].dt.day_name()
    return df
cat_features += ['S_2_day_of_week']

#%% FEATURE engineering NAGY
start_time = time.time()
train_data = train_data.to_pandas()
 
special = "S_19"  #x100 majd bin10


cols_to_step1 = ["D_108", "D_135", "D_137", "B_32", "B_33", "B_8", "D_103", "D_104",
                 "D_109", "D_112", "D_123", "D_125", "D_127", "D_128", "D_129",
                 "D_130", "D_131", "D_135", "D_137", "D_139", "D_140", "D_141",
                 "D_143", "D_54", "D_61", "D_63", "D_86", "D_92", "D_93", "D_94",
                 "D_96", "P_4", "R_12", "R_15", "R_19", "R_2", "R_21", "R_22",
                 "R_23", "R_24", "R_25", "R_27", "R_28", "R_4", "S_18", "S_20",
                 "S_6", "R_8", "B_41", "R_10", "R_11", "R_16", "R_20", "R_8",
                 "D_81", "S_19"]
cols_to_step05 = ["D_79", "B_22", "D_138", "D_78", "D_91"]
cols_to_step033 = ["D_107", "D_51"]
cols_to_step025 = ["D_44", "R_1", "D_136"]
cols_to_step02 = ["D_113"]
cols_to_step01 = ["D_89", "B_16", "B_16"]
cols_to_step005 = ["B_20", "D_122"]

cat_features += cols_to_step1
cat_features += cols_to_step05
cat_features += cols_to_step033
cat_features += cols_to_step025
cat_features += cols_to_step02
cat_features += cols_to_step01
cat_features += cols_to_step005


train_data = preprocess_df(train_data)

features = train_data.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
num_features = [col for col in train_data if col not in cat_features]
num_features = [col for col in train_data if col not in ['customer_ID', 'S_2']]

train_num_agg = train_data.groupby("customer_ID")[num_features].agg(['first', 'last'])
train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
train_num_agg.reset_index(inplace = True)
train_cat_agg = train_data.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
train_cat_agg.reset_index(inplace = True)
train_data = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_lbls, how = 'inner', on = 'customer_ID')
del train_num_agg, train_cat_agg


with open(f'train_data_aggregated.pkl','wb') as f:
    pickle.dump(train_data, f) 
del train_data 

test_data = pl.read_parquet('test_data.parquet')
test_data = preprocess_df(test_data.to_pandas())
print('Starting test feature engineer...')
test_num_agg = test_data.groupby("customer_ID")[num_features].agg(['first', 'last'])
test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
test_num_agg.reset_index(inplace = True)
test_cat_agg = test_data.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
test_cat_agg.reset_index(inplace = True)
test_data = test_num_agg.merge(test_cat_agg, how = 'inner', on = 'customer_ID')
del test_num_agg, test_cat_agg

with open(f'test_data_aggregated.pkl','wb') as f:
    pickle.dump(test_data, f) 



print("Futási idő:", time.time()-start_time, "s")
#%% FEATURE engineering2
start_time = time.time()
for cat_col in cat_features:
    encoder = LabelEncoder()
    train_data[cat_col] = encoder.fit_transform(train_data[cat_col])
    test_data[cat_col] = encoder.transform(test_data[cat_col])
print("Futási idő (LEBEL ENCODER):", int(time.time()-start_time), "s")    

 #%% ROUND Trick
start_time = time.time()
print("Kerekítés trükk indul!")
features = [col for col in train_data.columns if col not in ['customer_ID','S_2','S_2_max', CFG.target]]
features = [col for col in features if col not in cat_features]
# round trick
for col in features:
    if train_data[col].dtype=='float64':
        train_data[col] = train_data[col].astype('float32').round(decimals=2).astype('float16')
        test_data[col] = test_data[col].astype('float32').round(decimals=2).astype('float16')

print("Futási idő (Kerekítős trükk):", int(time.time()-start_time), "s") 

#%% TEST AND TRAIN SAVE

with open(f'train_data_rounded.pkl','wb') as f:
    pickle.dump(train_data, f) 
 
with open(f'test_data_rounded.pkl','wb') as f:
    pickle.dump(test_data, f)  
 

#%% Missing missing EDA új

background_color = 'white'
custom_colors = ["#ffd670","#70d6ff","#ff4d6d","#8338ec","#90cf8e"]
missing = pd.DataFrame(columns = ['% Missing values'],
                       data = train_data.isnull().sum()/len(train_data)*100)
unique = pd.DataFrame(columns = ['Unique'],data = train_data.nunique())
big_dataframe = pd.merge(missing, 
                         unique, 
                         left_index=True, 
                         right_index=True).sort_values(by=['% Missing values'])
corr = train_data.corr().loc[:, 'target'].round(decimals = 3).astype('string')

big_dataframe = pd.merge(big_dataframe, 
                         corr, 
                         left_index=True, 
                         right_index=True)
big_dataframe['labels'] = "Missing: " + big_dataframe['% Missing values'].astype('string') + " ,N_unique: " + big_dataframe['Unique'].astype('string')+" ,Corr: "+big_dataframe['target']

fig = plt.figure(figsize = (20, 60),facecolor=background_color)
gs = fig.add_gridspec(1, 2)
gs.update(wspace = 0.5, hspace = 0.5)
ax0 = fig.add_subplot(gs[0, 0])
for s in ["right", "top","bottom","left"]:
    ax0.spines[s].set_visible(False)

sns.heatmap(big_dataframe[['% Missing values']],cbar = False,annot = big_dataframe[['labels']] ,
            fmt ="", linewidths = 2,cmap = custom_colors,vmax = 1, ax = ax0)

plt.show()   





#%% KDE EDA minden oszlopra1

del_cols = [c for c in train_data.columns if (c.startswith(('D','t'))) & (c not in cat_features)]
df_del = train_data[del_cols]
spd_cols = [c for c in train_data.columns if (c.startswith(('S','t'))) & (c not in cat_features)]
df_spd = train_data[spd_cols]
pay_cols = [c for c in train_data.columns if (c.startswith(('P','t'))) & (c not in cat_features)]
df_pay = train_data[pay_cols]
bal_cols = [c for c in train_data.columns if (c.startswith(('B','t'))) & (c not in cat_features)]
df_bal = train_data[bal_cols]
ris_cols = [c for c in train_data.columns if (c.startswith(('R','t'))) & (c not in cat_features)]
df_ris = train_data[ris_cols]

#%% KDE EDA minden oszlopra: Distribution of Delinquency Variables
fig, axes = plt.subplots(29, 3, figsize = (40,120))
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(del_cols) - 1:
        sns.kdeplot(x = del_cols[i], hue='target', data = df_del, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
        ax.set_xlabel(del_cols[i] ,fontsize=30)
fig.suptitle('Distribution of Delinquency Variables', fontsize = 35, x = 0.5, y = 1)
plt.tight_layout()
plt.show()

#%% KDE EDA minden oszlopra: Distribution of Spend Variables
fig, axes = plt.subplots(8, 3, figsize = (16,18))
fig.suptitle('Distribution of Spend Variables', fontsize = 35, x = 0.5, y = 1)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(spd_cols) - 1:
        sns.kdeplot(x = spd_cols[i], hue ='target', data = df_spd, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
        ax.set_xlabel(spd_cols[i] ,fontsize=30)
plt.tight_layout()
plt.show()

#%% KDE EDA minden oszlopra: Distribution of Payment Variables
fig, axes = plt.subplots(1, 3, figsize = (12,4))
fig.suptitle('Distribution of Payment Variables',fontsize = 35)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(pay_cols) - 1:
        sns.kdeplot(x = pay_cols[i], hue ='target', data = df_pay, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        # sns.histplot(x = pay_cols[i], data = df_pay, ax = ax, alpha=0.1)

        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
        ax.set_xlabel(pay_cols[i] ,fontsize=30)
plt.tight_layout()
plt.show()

#%% KDE EDA minden oszlopra: Distribution of Balance Variables
fig, axes = plt.subplots(10, 4, figsize = (15,24))
fig.suptitle('Distribution of Balance Variables',fontsize = 15, x = 0.5, y = 1)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(bal_cols) - 1:
        sns.kdeplot(x = bal_cols[i], hue ='target', data = df_bal, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
        ax.set_xlabel(bal_cols[i] ,fontsize=30)
plt.tight_layout()
plt.show()

#%% KDE EDA minden oszlopra: Distribution of Risk Variables
fig, axes = plt.subplots(10, 3, figsize = (18,23))
fig.suptitle('Distribution of Risk Variables',fontsize=15, x = 0.5, y = 1)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(ris_cols) - 1:
        sns.kdeplot(x = ris_cols[i], hue ='target', data = df_ris, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
        ax.set_xlabel(ris_cols[i] ,fontsize=30)
plt.tight_layout()
# plt.show()
plt.savefig("Distribution of Risk Variables.png")



plt.show()   



#%% FEATURE engineering DEF

def fill_nan_9(df, col):
    df[col].fillna(value = 9,
               inplace = True,
              # axis = 1,
              )   

bin_to_step_1 = [-0.5]
step = 1
for i in range(1,52,1):
    bin_to_step_1.append(bin_to_step_1[-1]+step)
    
bin_to_step_05 = [-0.25]
step = 0.5
for i in range(1,100,1):
    bin_to_step_05.append(bin_to_step_05[-1]+step)

bin_to_step_033 = [-0.33/2]
step = 0.33
for i in range(1,150,1):
    bin_to_step_033.append(bin_to_step_033[-1]+step)
    
bin_to_step_25 = [-0.125]
step = 0.25
for i in range(1,200,1):
    bin_to_step_25.append(bin_to_step_25[-1]+step)
    
bin_to_step_02 = [-0.1]
step = 0.2
for i in range(1,250,1):
    bin_to_step_02.append(bin_to_step_02[-1]+step)

bin_to_step_01 = [-0.05]
step = 0.1
for i in range(1,400,1):
    bin_to_step_01.append(bin_to_step_01[-1]+step)
    
bin_to_step_005 = [-0.025]
step = 0.05
for i in range(1,800,1):
    bin_to_step_005.append(bin_to_step_005[-1]+step)

def transform_to_step1(df, col):
    df[col] = np.digitize(df[col], bin_to_step_1)

def transform_to_step05(df, col):
    df[col] = np.digitize(df[col], bin_to_step_05)

def transform_to_step033(df, col):
    df[col] = np.digitize(df[col], bin_to_step_033)
    
def transform_to_step025(df, col):
    df[col] = np.digitize(df[col], bin_to_step_25)   
    
def transform_to_step02(df, col):
    df[col] = np.digitize(df[col], bin_to_step_02)

def transform_to_step01(df, col):
    df[col] = np.digitize(df[col], bin_to_step_01) 
    
def transform_to_step005(df, col):
    df[col] = np.digitize(df[col], bin_to_step_005) 

def S_19_special(df):
    df["S_19"] = df["S_19"] * 100
    df["S_19"] = np.digitize(df["S_19"], bin_to_step_1)        
 
def drop_rows_higher_than_1_1(df, col):
    df = df[df[col] < 1.1]
 
nem_tudom_mit_csinaljak_vele = ["R_26"]  


#%% FILL nan 9
start_time = time.time()
cols_to_fill_nan_9 = ["D_108", "D_135", "D_137", "D_138", "D_81"]
def lets_fill_nan_9(df):
    for col in cols_to_fill_nan_9:
        fill_nan_9(df, col)
        
lets_fill_nan_9(train_data)
lets_fill_nan_9(test_data)


#%% Binning TRAIN data
S_19_special(train_data)
S_19_special(test_data)
#Train data:
for col in cols_to_step1:
    transform_to_step1(train_data, col)
    
for col in cols_to_step05:
    transform_to_step05(train_data, col)

for col in cols_to_step033:
    transform_to_step033(train_data, col)
    
for col in cols_to_step025:
    transform_to_step025(train_data, col)
    
for col in cols_to_step02:
    transform_to_step02(train_data, col)
    
for col in cols_to_step01:
    transform_to_step01(train_data, col)
    
for col in cols_to_step005:
    transform_to_step005(train_data, col)

#%% Binning test data

for col in cols_to_step1:
    transform_to_step1(test_data, col)
    
for col in cols_to_step05:
    transform_to_step05(test_data, col)

for col in cols_to_step033:
    transform_to_step033(test_data, col)
    
for col in cols_to_step025:
    transform_to_step025(test_data, col)
    
for col in cols_to_step02:
    transform_to_step02(test_data, col)
    
for col in cols_to_step01:
    transform_to_step01(test_data, col)
    
for col in cols_to_step005:
    transform_to_step005(test_data, col)
#%% SAVE DATA BINNED

with open(f'train_data_binned.pkl','wb') as f:
    pickle.dump(train_data, f) 
 
with open(f'test_data_binned.pkl','wb') as f:
    pickle.dump(test_data, f)  


    
#%% COLS to DROP     
cols_to_drop = ["B_39", "B_42", "D_110", "D_111", "D_132", "D_134", "D_142",
                "D_46", "R_26"]

#%% Print info for testing

def make_info(col):
    print(train_data[col].describe())
    train_data[col].hist(bins=50)
    plt.show()
    print(sorted(train_data[col].unique()))
    print("1.1 fölött:", train_data[train_data[col] > 1.1 ][col].sum())
# make_info("D_122")

print("Futási idő (FILTER és egyéb):", int(time.time()-start_time), "s") 
 
#%% BETANÍTÁS    LGBM 

features = [col for col in train_data.columns if col not in ['customer_ID','S_2','S_2_max', CFG.target]]
params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': CFG.seed,
    'num_leaves': 100,
    'learning_rate': 0.005,
    'feature_fraction': 0.20,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'n_jobs': -1,
    'lambda_l2': 2,
    'min_data_in_leaf': 40
    }

# Create a numpy array to store out of folds predictions
oof_predictions = np.zeros(len(train_data))
kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_data, train_data[CFG.target])):
    print(' ')
    print('-'*50)
    print(f'Training fold {fold} with {len(features)} features...')
    x_train, x_val = train_data[features].iloc[trn_ind], train_data[features].iloc[val_ind]
    y_train, y_val = train_data[CFG.target].iloc[trn_ind], train_data[CFG.target].iloc[val_ind]
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature = cat_features)
    lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature = cat_features)
    model = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 200,
        valid_sets = [lgb_train, lgb_valid],
        early_stopping_rounds = 100,
        verbose_eval = 100,
        feval = lgb_amex_metric
        )
    # Save best model
    joblib.dump(model, f'lgbm_fold{fold}_seed{CFG.seed}.pkl')
    # Predict validation
    val_pred = model.predict(x_val)
    # Add to out of folds array
    oof_predictions[val_ind] = val_pred
    # Predict the test set
    # test_pred = model.predict(test_data[features])
    # test_predictions += test_pred / CFG.n_folds
    # Compute fold metric
    score = amex_metric(y_val, val_pred)
    print(f'Our fold {fold} CV score is {score}')
    del x_train, x_val, y_train, y_val, lgb_train, lgb_valid

# Compute out of folds metric
score = amex_metric(train_data[CFG.target], oof_predictions)
print(f'Our out of folds CV score is {score}')
# Create a dataframe to store out of folds predictions
oof_df = pd.DataFrame({'customer_ID': train_data['customer_ID'], 
                       'target': train_data[CFG.target], 
                       'prediction': oof_predictions})
oof_df.to_csv(f'oof_lgbm_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
# Create a dataframe to store test prediction
# test_df = pd.DataFrame({'customer_ID': test_data['customer_ID'], 
#                         'prediction': test_predictions})
# test_df.to_csv(f'test_lgbm_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)



#%% Catboos Eval metric

class CatBoostEvalMetricAmex(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # the larger metric value the better
        return True

    def evaluate(self, approxes, target, weight):
        return amex_metric(target, approxes[0]), 1



#%% BETANÍTÁS CATBOOST
start_time = time.time()
"""
model = CatBoostClassifier(iterations=1000,
                           task_type="GPU",
                           devices='0:1')

<unit ID> for one device (for example, 3)
<unit ID1>:<unit ID2>:..:<unit IDN> for multiple devices (for example, devices='0:1:3')
<unit ID1>-<unit IDN> for a range of devices (for example, devices='0-3')
"""

features = [col for col in train_data.columns if col not in ['customer_ID','S_2','S_2_max', CFG.target]]
params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': CFG.seed,
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.20,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'n_jobs': -1,
    'lambda_l2': 2,
    'min_data_in_leaf': 40
    }

N_FOLDS = 2
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=22)
y_oof = np.zeros(train_data.shape[0])
# y_test = np.zeros(test.shape[0])
ix = 0

# train_x = train_data[features]
# train_y = train_data[CFG.target]
train_x = train_data[features]
train_y = train_data[CFG.target]
for train_ind, val_ind in skf.split(train_x, train_y):
    print(f"******* Fold {ix} ******* ")
    tr_x, val_x = (
        # train_x.iloc[train_ind].reset_index(drop=True),
        # train_x.iloc[val_ind].reset_index(drop=True),
        train_x.iloc[train_ind],
        train_x.iloc[val_ind],
    )
    tr_y, val_y = (
        # train_y.iloc[train_ind].reset_index(drop=True),
        # train_y.iloc[val_ind].reset_index(drop=True),
        train_y.iloc[train_ind],
        train_y.iloc[val_ind],
    )

    clf = CatBoostClassifier(iterations=10, random_state=50)
    clf.fit(tr_x, tr_y, eval_set=[(val_x, val_y)], cat_features=cat_features,  verbose=50)
    preds = clf.predict_proba(val_x)[:, 1]
    y_oof[val_ind] = y_oof[val_ind] + preds

    # preds_test = clf.predict_proba(test)[:, 1]
    # y_test = y_test + preds_test / N_FOLDS
    ix = ix + 1
y_pred = train_y.copy(deep=True)
# y_pred = y_pred.rename({"target": "prediction"})
# y_pred.columns = ["prediction"]
# y_pred["prediction"] = y_oof
val_score = amex_metric(train_y, y_oof)
print(f"Amex metric: {val_score}")
print("Futási idő (Catboost):", int(time.time()-start_time)/60, "min") 

# LGBM minden nélkül: 0.6683 - amex1.py
# LGBM cat col label encoder + s_2 kezelése : 0.752 - amex1.py
# LGBM cat col label encoder + s_2 kezelése : 0.752 - amex2.py (i=200, lr = 0.005)
# CAT BOOST cat col label encoder + s_2 kezelése : 0.7806 - amex2.py
# CAT BOOST cat col label encoder + s_2 kezelése : 0.7817 - amex2.py (i=400)
# CAT BOOST cat col label encoder + s_2 kezelése + round : 0.7826 - amex2.py (i=400)
# LGBM cat col label encoder + s_2 kezelése + round + bonyi kicsit : 0.752 - amex2.py (i=200, lr = 0.005)
# CAT BOOST cat col label encoder + s_2 kezelése + round + bonyi kicsit: 0.7838 - amex2.py (i=600)
# CAT BOOST  + s_2 kezelése + round + bonyi kész: 0.7827 - amex3.py (i=600)
# CAT BOOST  + s_2 kezelése + round + bonyi kész + dropna: 0.7482 - amex3.py (i=100)
# CAT BOOST  + s_2 kezelése + round + bonyi kész + fill na median and extra: 0.77834 - amex3.py (i=100)
# CAT BOOST  + s_2 kezelése + round + bonyi kész + fill na median and extra: 0.78311 - amex3.py (i=1000)
# CAT BOOST  + s_2 kezelése + round + bonyi kész + dropcol nélkül + fill na median and extra: 0.7791 - amex3.py (i=100)
# CAT BOOST  + s_2 kezelése + round + bonyi kész + összes dropcol nélkül + fill na median and extra col: 0.7840 - amex3.py (i=100)
# CAT BOOST  + s_2 kezelése + round + bonyi kész + összes dropcol nélkül + fill na median and extra col: 0.7870 - amex3.py (i=1000)



#%% Impute nan median
train_data_nan_median = train_data.copy()

def impute_nan_median(df, variable, median):
    df[variable+"_median"] = df[variable].fillna(median)
    
    
def capture_nan_and_fillna(df, variable):
    df[variable+"_NAN"] = np.where(df[variable].isnull(),1,0)
    df[variable] = df[variable].fillna(df[variable].median())
    
features_with_missing_rows =big_dataframe[ big_dataframe['% Missing values'] > 0 ].index 
for col in features_with_missing_rows:
    capture_nan_and_fillna(train_data_nan_median, col)   

#%% Variance filter
# Variance filter
features_variance_filter = [col for col in train_data.columns if col not in ['customer_ID','S_2','S_2_max', 'target']]
train_data_to_filter = train_data[features_variance_filter]
n_elotte = train_data_to_filter.shape[1]
# Removing all features that have variance under 0.05
selector = VarianceThreshold(threshold = 0.05)
selector.fit(train_data_to_filter)
mask_clean = selector.get_support()
train_data_filtered = train_data_to_filter[train_data_to_filter.columns[mask_clean]]
# df_test = df_test[df_test.columns[mask_clean]]
n_utana = train_data_filtered.shape[1]
print("Variance filter után kitörölt oszlopok száma:", n_elotte-n_utana)


#%% Oszlop kiválasztó definíciók és Reverse selection

def metric_measure(variables, df_target, df):
    X = df.loc[:,variables]
    y = df_target
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=global_random)
    model = xgboost.XGBRegressor(n_estimators=100)
    # print("Első kezd.")
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    score = absolute(scores).mean()
    # print("Első végez:", score)
    return score

def next_best(current_variables,candidate_variables, df_target, df):
    best_score = -1
    best_variable = None
    candidate_variables_tqdm = tqdm(candidate_variables)
    for v in candidate_variables_tqdm:
        score_v = metric_measure(current_variables + [v], df_target, df)
        if score_v >= best_score:
            best_score = score_v
            best_variable = v
    print('Aktuális legjobb a',best_variable, 'pontossága:',best_score)
    return best_variable, best_score

def next_worst(current_variables,candidate_variables, df_target, df):
    worst_score = 0
    worst_variable = None
    candidate_variables_tqdm = tqdm(candidate_variables)
    for v in candidate_variables_tqdm:
        candidate_variables = [can for can in candidate_variables if can != v]
        variables = current_variables + candidate_variables
        score_v = metric_measure(variables, df_target, df)
        if worst_score == 0:
            worst_score = score_v
        if score_v <= worst_score:
            worst_score = score_v
            worst_variable = v
    print('Aktuális legjobb a',worst_variable, 'pontossága:',worst_score)
    return worst_variable, worst_score

iternum = []
scores = []
for i in tqdm(range(0,max_number_variables)):
    print('Új kör kezdődik.')
    s = time.time()
    worst_var, best_score = next_best(current_variables,candidate_variables,df_train_target,df_train)
    # current_variables = current_variables + [next_var]
    candidate_variables.remove(worst_var)
    iternum.append(len(current_variables))
    scores.append(best_score)
    e = time.time()
    print("Betanítás 1 feature = {}".format(e-s))
    print(len(scores),'Végzett a', worst_var)
    print("Eddigi oszlopok:", current_variables)
print(current_variables)
  
