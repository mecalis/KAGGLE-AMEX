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
~ 12 categorical cols were officially predefined.
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
<img src="missing4.png" width="600">

3. EDA

```
```

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


4. 
