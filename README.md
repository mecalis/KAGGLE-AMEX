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

2. EDA
4. 
