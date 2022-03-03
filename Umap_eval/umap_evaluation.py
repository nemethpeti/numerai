# import dependencies
import pandas as pd
import json
import xgboost
import gc

from cuml import UMAP

pd.options.mode.chained_assignment = None  # default='warn'

numeraiDir = '.'
import numeraiutils as num

target='target'
keys = ['era', 'data_type', target]

# In[ load data ]
print('Loading data')

f = open(f'{numeraiDir}/numerai_datasets/features.json')
j = json.load(f)

train = pd.read_parquet(f'{numeraiDir}/numerai_datasets/numerai_training_data.parquet')
val = pd.read_parquet(f'{numeraiDir}/numerai_datasets/numerai_validation_data.parquet')
td = pd.read_parquet(f'{numeraiDir}/numerai_datasets/numerai_tournament_data.parquet')

# all features
# features = [c for c in train if c.startswith("feature")]

# selected features for speed
features = j['feature_sets']['medium'] # small / medium / legacy

targets = [c for c in train if c.startswith("target")]

train = train[keys + features]
val = val[keys + features]
td = td[keys + features]

# In[ baseline model ]

xgb_params = {
    "n_estimators" : 1000,
    "max_depth" : 5,
    "learning_rate" : 0.01,
    "colsample_bytree" : 0.1,
    "tree_method" : 'gpu_hist'
    }
    
est = xgboost.XGBRegressor(**xgb_params)

est.fit(train[features], train[target])
val['pred'] = est.predict(val[features])

num.evaluation(val, features)
val = val.rename(columns={'pred':'prediction'})
val['prediction'].to_csv('validation_baseline.csv')

# Numerai score (spearman correlation): 0.0208, STD: 0.0329
# Sharpe ratio: 0.6320
# Sortino ratio: 1.6578
# Max drawdown: -0.19033823759034238
# Max feature exposure:  0.4509352785060702

# In[ fit umap model with full dataset that includes validation and tournament data]

n_neighbors = 15
min_dist = 0
n_components = 60

train_full = pd.concat([train, val, td[::10]]).reset_index()
train_tr = pd.concat([train, val]).reset_index()


umap = UMAP(n_neighbors = n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
data = umap.fit_transform(train_full[features])

umap_features = [f'f_{i}' for i in range(data.shape[1])]

pData = pd.DataFrame(data, columns=umap_features).iloc[:len(train_tr)]
new_train = pd.concat([train_tr, pData], axis=1)    
new_features = features + umap_features

del train_full
del train_tr
del umap
del data
del pData

gc.collect()


# In[ retrain estimator on the complete extended dataset ]

est = xgboost.XGBRegressor(**xgb_params)
est.fit(new_train.loc[new_train.data_type=='train', new_features], new_train.loc[new_train.data_type=='train', target])
new_train.loc[new_train.data_type=='validation', 'pred'] = est.predict(new_train.loc[new_train.data_type=='validation', new_features])
num.evaluation(new_train[new_train.data_type=='validation'], new_features)

val['prediction'] = new_train.loc[new_train.data_type=='validation', 'pred'].values
val['prediction'].to_csv('validation_allFeatures.csv')

# Numerai score (spearman correlation): 0.0213, STD: 0.0329
# Sharpe ratio: 0.6495
# Sortino ratio: 1.7063
# Max drawdown: -0.20478065308905238
# Max feature exposure:  0.4371256652128962


# In[ retrain estimator on the extended dataset only ]

est = xgboost.XGBRegressor(**xgb_params)
est.fit(new_train.loc[new_train.data_type=='train', umap_features], new_train.loc[new_train.data_type=='train', target])
new_train.loc[new_train.data_type=='validation', 'pred'] = est.predict(new_train.loc[new_train.data_type=='validation', umap_features])
num.evaluation(new_train[new_train.data_type=='validation'], features)

val['prediction'] = new_train.loc[new_train.data_type=='validation', 'pred'].values
val['prediction'].to_csv('validation_umapFeatures.csv')

# Numerai score (spearman correlation): 0.0073, STD: 0.0158
# Sharpe ratio: 0.4651
# Sortino ratio: 0.9302
# Max drawdown: -0.10544801045624015
# Max feature exposure:  0.07221613666926133




