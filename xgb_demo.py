
import numeraiutils as num
import numpy as np

import joblib
from xgboost import XGBRegressor

# In[Read data]:

df = num.read_csv("../numerai_training_data.csv")

features = [c for c in df if c.startswith("feature")]
df["erano"] = df.era.str.slice(3).astype(int)
print(len(features), len(df.erano.unique()))

td = num.read_csv('../numerai_tournament_data.csv')
td['id'] = td.index.values
val = td[td.data_type=='validation']
val["erano"] = val.era.str.slice(3).astype(int)

print(len(val.erano.unique()))

train=df.copy()
train = train.append(val)


# In[Simple XGB Model]:

    
class xgbModel():
    def __init__(self, features, **kwargs):
        
        self.features = features
        self.model = XGBRegressor(**kwargs)
        
    def fit(self, trainSet):
        
        self.model.fit(trainSet[self.features], trainSet['target'])
          
    def evaluate(self, testSet, neutralize=0):
        
        testSet['pred'] = self.model.predict(testSet[self.features])
        
        if neutralize > 0:
            num.neutralize(testSet, neutralize, self.features)
        
        corr, std = num.numerai_score(testSet)
        max_drawdown = num.drawdown(testSet)
        
        res = {'corr':corr, 'std':std, 'sharpe':corr/std, 'drawdown':max_drawdown}
        return str(res).replace("'", '"')


# In[Cross validation with the XGB Model]:

xgb = num.CrossVal(train, xgbModel, features, colsample_bytree=0.1, learning_rate=0.01, max_depth=5, n_estimators=2000, random_state=42, n_jobs=-1, tree_method='gpu_hist', gpu_id=0)
corr, std, sharpe, max_drawdown = xgb.run(save_file='XGB', neutralize=0, save_model=True)

# In[Cross Validation with neutralization]:

for n in np.arange(0, 1, 0.1):

    xgb = num.CrossVal(train, xgbModel, features, colsample_bytree=0.1, learning_rate=0.01, max_depth=5, n_estimators=2000, random_state=42, n_jobs=-1, tree_method='gpu_hist', gpu_id=0)
    corr, std, sharpe, max_drawdown = xgb.run(save_file='XGB_'+str(n), neutralize=n)


# In[Run prediction on tournament data]:

td.loc[:, 'prediction']=0
for m in range(15):
         
         f = f'./CV_model_{m}-XGB_tuned.model'
         print(f)
         loaded_model = XGBRegressor()
         loaded_model = joblib.load(f)

         td.loc[:, 'prediction'] += loaded_model.predict(td[features])
         
         del f
         del loaded_model
         
td['prediction'] /= 15

num.submit(td, 'model_name')