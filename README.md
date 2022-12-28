# KAGGLE-AMEX
Kaggle Amex competition in 2022 Summer

## INFO:
- I took part in this contest in my exiguous free time (when my children fall asleep after 10 pm), so this project is not full
- In phase of aggregation I lost more than 2% accuracy besause of lack of memory. The size of traning and test data is more than 25GB, so I coud do just the basic aggregation after groupby. Top10 contestens created more than 5k new columns after aggregation (for example lag features)   
- The winner achived 0.8+ accuracy.
- I achived 0.78 accuracy (3000. place from 5000)

## The dataset

American express credit card dataset.
The goal is to "guess" which customer can not pay back his credit card debt.
The training set contains 1M+ rows and 200+ anonymized columns.

## My solution
1. Basic preprocessing and aggregation
~ 12 categorical cols were officially predefined. Here I lost the chance to enter the top5% because of lack of computer memory (lag features)
```
train_num_agg = train_data.groupby("customer_ID")[num_features].agg(['first', 'last'])
train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
train_num_agg.reset_index(inplace = True)
train_cat_agg = train_data.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
train_cat_agg.reset_index(inplace = True)
train_data = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_lbls, how = 'inner', on = 'customer_ID')
del train_num_agg, train_cat_agg
```
Labelencoding:
```
for cat_col in cat_features:
    encoder = LabelEncoder()
    train_data[cat_col] = encoder.fit_transform(train_data[cat_col])
    test_data[cat_col] = encoder.transform(test_data[cat_col])
```
2. Missing EDA

There are couple completly useless or highly correlated columns.
```
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
```
Result:
After the round trick I realised there are 30+ other, hidden categorical columns.
<img src="images/missing4.png" width="800">

3. Round trick
Both the competitors and I noticed that randon noise where added to the data.
```
features = [col for col in train_data.columns if col not in ['customer_ID','S_2','S_2_max', CFG.target]]
features = [col for col in features if col not in cat_features]
# round trick
for col in features:
    if train_data[col].dtype=='float64':
        train_data[col] = train_data[col].astype('float32').round(decimals=2).astype('float16')
        test_data[col] = test_data[col].astype('float32').round(decimals=2).astype('float16')
```


4. EDA

```
#%% KDE EDA minden oszlopra

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
```
```
#%% KDE EDA minden oszlopra: Distribution of Payment Variables
fig, axes = plt.subplots(1, 3, figsize = (12,4))
fig.suptitle('Distribution of Payment Variables',fontsize = 35)
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(pay_cols) - 1:
        sns.kdeplot(x = pay_cols[i], hue ='target', data = df_pay, fill = True, ax = ax, palette =["#e63946","#8338ec"])
        ax.tick_params()
        ax.xaxis.get_label()
        ax.set_ylabel('')
        ax.set_xlabel(pay_cols[i] ,fontsize=30)
plt.tight_layout()
plt.show()
```

<img src="images/Distribution of Payment Variables.png" width="800">

5. Feature engineering: binning

I realised that there are binned columns, for example [0,1,3,4,5...] or [0.1, 0.2, 0.3....] but there are hidden because of the random noise. These part of the script is completelly my own solution (perhaps its effectiveness is questionable, there was no time to test is thoroughly) 
```
#%% FEATURE engineering DEF

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
```

