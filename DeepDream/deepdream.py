import tensorflow as tf
import xgboost

import numpy as np
import pandas as pd

import json
import sys

root =  'c:/'
numeraiDir = '/Users/Newton/Documents/Trading/Numerai'
sys.path.append(f'{root}{numeraiDir}/')
import numeraiutils as num


target='target'
keys = ['era', 'data_type', 'target']

# In [ load data ]
print('Loading data')

f = open(f'{root}/Users/Newton/Documents/Trading/Numerai/numerai_datasets/features.json')
j = json.load(f)

train = pd.read_parquet(f'{root}/{numeraiDir}/numerai_datasets/numerai_training_data.parquet')
val = pd.read_parquet(f'{root}/{numeraiDir}/numerai_datasets/numerai_validation_data.parquet')

# select features set
#features = j['feature_stats'].keys()
features = j['feature_sets']['medium'] # small / medium / legacy
n_features = len(features)

targets = [c for c in train if c.startswith("target")]

# reduce data set size
# eras = train.era.unique()[::4]
# train = train[train.era.isin(eras)]

# In[dream ]

base_model = tf.keras.models.load_model('bestModel0239-16-8-fullSet-Dropout.h5')
base_model.compile(run_eagerly=True)

# Maximize the activations of these layers
name = 'concat'
layers = base_model.get_layer(name).output

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

"""## Calculate loss

The loss is the sum of the activations in the chosen layers. 
The loss is normalized at each layer so the contribution from larger layers does not outweigh smaller layers. 
Normally, loss is a quantity we wish to minimize via gradient descent. 
In DeepDream, we maximize this loss via gradient ascent.
"""
class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[n_features], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  
  def __call__(self, batch, steps, step_size):
    print("Tracing")
    loss = tf.constant(0.0)
    
    for n in tf.range(steps):
      with tf.GradientTape() as tape:
        # This needs gradients relative to the input row
        # `GradientTape` only watches `tf.Variable`s by default
        tape.watch(batch)
        
        layer_activations = self.model(batch)
        
        # calculate loss based on selected neurons
        loss = tf.math.reduce_mean(layer_activations, -1)
      
      # Calculate the gradient of the loss with respect to the pixels of the input image.
      gradients = tape.gradient(loss, batch)
      
      # Normalize the gradients.
      gradients /= tf.expand_dims(tf.math.reduce_std(gradients, -1), 1) + 1e-8 
      
      # In gradient ascent, the "loss" is maximized so that the input row increasingly "excites" the layers.
      # You can update the tow by directly adding the gradients (because they're the same shape!)
      batch = batch + gradients*step_size


    batch = tf.clip_by_value(batch, 0, 1)
    
    return batch


tf.config.run_functions_eagerly(True)
deepdream = DeepDream(dream_model)

# test run
rec = np.array([train.iloc[0][features].astype(np.float32).values])
dream = deepdream(tf.convert_to_tensor(rec), steps=5, step_size=0.01)
print(np.max(rec-dream), np.std(rec-dream))

# In[ execute ]

data = train.reset_index()
step_size=0.01
steps=5

dreamdf = pd.DataFrame()
batch_size=200000

for i in np.arange(0, len(data), batch_size):
    print(i)
    
    start = i
    end = np.minimum(i+batch_size-1, len(data)-1)
    print(start, end)

    batch = tf.convert_to_tensor(data.loc[start:end, features].astype(np.float32).values)
     
    
    dream = deepdream(batch, steps, step_size)
    
    newdf = pd.DataFrame(dream, columns=features)
    newdf[target] = data.loc[start:end, target].values


    dreamdf = pd.concat([dreamdf, newdf])


dreamdf.to_parquet('dream_train.prq')


# In[ evaluate ]

#dreamdf=pd.read_parquet('dream_train.prq')
#dreamdf = dreamdf.sample(frac = 0.05, random_state=42)
#new_train = pd.concat([train, dreamdf])


# baseline
# Numerai score (spearman correlation): 0.0198, STD: 0.0336
# Sharpe ratio: 0.5883
# Sortino ratio: 1.4523
# Max drawdown: -0.24000229180259036

# baseline full dataset medium 2000
# Numerai score (spearman correlation): 0.0211, STD: 0.0312
# Sharpe ratio: 0.6769
# Sortino ratio: 1.8660
# Max drawdown: -0.1800844114038771
# Max feature exposure:  0.42000062702669405

# baseline full dataset medium 1000
# Numerai score (spearman correlation): 0.0209, STD: 0.0329
# Sharpe ratio: 0.6338
# Sortino ratio: 1.7049
# Max drawdown: -0.2030559191107792
# Max feature exposure:  0.44000001329856325


def getBaseLine(train, name):
    xgb_params = {
        "n_estimators" : 1000,
        "max_depth" : 5,
        "learning_rate" : 0.01,
        "colsample_bytree" : 0.1,
        "random_state" : 42,
        "tree_method" : 'gpu_hist',
        'n_jobs' : -1
        }
    
    print('Fitting Estimator - ', name)
    est = xgboost.XGBRegressor(**xgb_params)
    est.fit(train[features], train[target])
    
    val['pred'] = est.predict(val[features])
    num.evaluation(val, features)
    
    val.rename(columns={'pred':'prediction'})['prediction'].to_csv(name)
    
#getBaseLine(train.sample(frac=1), 'baseline.csv')        

# concat, 2000, full dataset
# Numerai score (spearman correlation): 0.0220, STD: 0.0311
# Sharpe ratio: 0.7066
# Sortino ratio: 2.0523
# Max drawdown: -0.17057973703073878
# Max feature exposure:  0.4170446375539873





for name in ["", '--3', '12-21']:
    dreamdf=pd.read_parquet(f'dream_train{name}.prq')    
    dreamdf = dreamdf.sample(frac = 0.05, random_state=42)
    new_train = pd.concat([train, dreamdf])
    
    getBaseLine(new_train, f'new_train_{name}.csv')
    
# In[ combined]    

new_train=train.copy()
for name in ['concat', 'batch_normalization_42']:
    dreamdf=pd.read_parquet(f'dream_train_{name}.prq')    
    dreamdf = dreamdf.sample(frac = 0.025, random_state=42)
    new_train = new_train.append(dreamdf)
    
getBaseLine(new_train, 'new_train_sum.csv')
