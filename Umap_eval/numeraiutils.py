# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:07:06 2021

@author: Newton
"""

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import itertools
import joblib
import gc
import numerapi
#import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
import scipy
import xgboost


# =============================================================================
# supporting function
# =============================================================================

target = "target"
pred = 'pred'
era = 'era'

val_era1 = ['era121', 'era122', 'era123', 'era124', 'era125', 'era126', 'era127', 'era128', 'era129', 'era130', 'era131', 'era132']
val_era2 = ['era197', 'era198', 'era199', 'era200', 'era201', 'era202', 'era203', 'era204', 'era205', 'era206', 'era207', 'era208', 'era209', 'era210', 'era211', 'era212']
val_eras = val_era1 + val_era2

def submit(df, model_name, filename='predictions.csv', version=1):
    
    if 'id' not in df.columns:
        df['id'] = df.index

    if 'prediction' not in df.columns:
        df['prediction'] = df['pred']
        
    df[['id', 'prediction']].to_csv(filename, index=False)
    
    public_id = "xx"
    secret_key = "xx"
    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
        
    models = napi.get_models()
    model_id = models[model_name]
        
    napi.upload_predictions(filename, model_id=model_id, version=version)


def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]

# convenience method for scoring
def score(df):
    return correlation(df[pred], df[target])    

def numerai_score(df):
    scores = df.groupby('era').apply(score)
    return scores.mean(), scores.std(ddof=0)
    
def plot_scores(df):
    scores = df.groupby('era').apply(score)
    plt.xticks(rotation='vertical')
    plt.bar(df.era.unique(), scores)
    
def sharpe(df):

    scores = df.groupby('era').apply(score)

    sharpe = scores.mean() / scores.std(ddof=0)
    
    sortino = 0
    if np.sum(np.minimum(0, scores))!=0:
        sortino = scores.mean() / (np.sum(np.minimum(0, scores)**2)/(len(scores)))**.5
        
    return sharpe, sortino
  
def drawdown(df):
    
    scores = df.groupby('era').apply(score)
    
    rolling_max = (scores + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (scores + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    
    return max_drawdown
    

def feature_exposure(df, features):
    # Check the feature exposure of your validation predictions
    max_per_era = df.groupby("era").apply(lambda d: d[features].corrwith(d[pred]).abs().max())
    print("Max feature exposure: ", max_per_era.mean()) 
    
    return max_per_era.mean()

    
def evaluation(df, features):    
    mean, std = numerai_score(df)
    print("Numerai score (spearman correlation): {:0.4f}, STD: {:0.4f}".format(mean, std))
    #payout(df)
    s,sort = sharpe(df)
    print("Sharpe ratio: {:0.4f}".format(s))
    print("Sortino ratio: {:0.4f}".format(sort))

    max_drawdown = drawdown(df)
    print(f"Max drawdown: {max_drawdown}")    

    feature_exposure(df, features)
    plot_scores(df)
    print()
    
def fast_evaluation(df, features):    
    mean, std = numerai_score(df)
    print("Numerai score (spearman correlation): {:0.4f}, STD: {:0.4f}".format(mean, std))
    #payout(df)
    s,sort = sharpe(df)
    print("Sharpe ratio: {:0.4f}".format(s))
    print("Sortino ratio: {:0.4f}".format(sort))

    max_drawdown = drawdown(df)
    print(f"Max drawdown: {max_drawdown}")    

    print()    
    
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)
    return df   

def read_data():
    
    df = pd.read_parquet("C:\\Users\\Newton\\Documents\\Trading\\Numerai\\training.prq")
    df["erano"] = df.era.str.slice(3).astype(int)
    
    val = pd.read_parquet('C:\\Users\\Newton\\Documents\\Trading\\Numerai\\validation.prq')
    val["erano"] = val.era.str.slice(3).astype(int)
    
    features = [c for c in df if c.startswith("feature")]
    
    return df, val, features

