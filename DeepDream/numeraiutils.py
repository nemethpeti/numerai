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
    
    public_id = "x"
    secret_key = "x"
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



def _neutralize(df, columns, by, proportion=1.0):
    scores = df[columns]
    exposures = df[by].astype('float32').values
    scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))
    return scores / scores.std(ddof=0)
def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)
def normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, proportion)
    return df[columns]

def neutralize(df, rate, predict_features):

    
    df["pred"] = df.groupby("era").apply(
        lambda x: normalize_and_neutralize(x, ['pred'], predict_features, rate)
    )

    df["pred"] = MinMaxScaler().fit_transform(df[["pred"]]) # transform back to 0-1
    
    return df

def MDA(model, features, testSet, iterations = 5):
    
    testSet['pred'] = model.predict(testSet[features])
    corr, std = numerai_score(testSet)
    print("Base corr: ", corr)
    diff = []
    #np.random.seed(seed)
    for col in features:

        X = testSet.copy()
        corrSum = 0
        
        for i in range(iterations):
            np.random.shuffle(X[col].values)
            testSet['pred'] = model.predict(X[features])
            corrX, stdX = numerai_score(testSet)
            corrSum += corrX
        
        corrSum /= iterations
        print(col, corrSum-corr)
        diff.append((col, corrSum-corr))
        
    return diff


class CrossVal():
    
    def __init__(self, df, model, features, N = 6, k = 2, purge = 8, embargo = 4, target='target', **kwargs):
        
        print('v6')
        self.df = df
        
        self.model = model(features, target=target, **kwargs)
        self.combinations = list(itertools.combinations(range(1, N+1), k))      
        self.fi = int(k/N * len(self.combinations))
        
        self.eras = self.df.era.unique().astype('int')
        self.N = N
        self.k = k
        self.purge = purge
        self.embargo = embargo
        self.features = features
        self.target = target
        
        self.results = pd.DataFrame(index = [ 'split'+str(i) for i in range(1, self.N+1)], columns = ['path'+str(i) for i in range(1, self.fi+1)] )
        
        # save splits
        
        #eraNo = 120
        eraNo = len(df.era.unique())
        
        size = eraNo // self.N
        self.valSplits = [range(i*size+1, (i+1)*size+1) for i in range(self.N)]
            
        
        print(self.valSplits)
        #print(self.fi)
        #print(self.results)
        #print(self.combinations)

        
    def split_data(self, combination):
        
        tests = []
        test = []
        banned = []        
        
        for split in combination:
            test_split = self.valSplits[split - 1]
            tests.append( list(test_split) )
            test += test_split
            banned += range(test_split[0] - self.purge , test_split[0] )
            banned += range(test_split[-1], test_split[-1] + self.purge + self.embargo + 1 )
            
        train = sorted(list(set(self.eras).difference(set(test).union(set(banned)))))
        
        return train, sorted(tests)
    
    def run(self, save_file=None, run_from=0, save_model=False, neutralize=0):
        
        pathCounter = {}
        for i in range(self.N):
            pathCounter.update({'split'+str(i+1): 0}) 
        
        for com_idx, test_splits in enumerate(self.combinations[run_from:]):
            
            print(test_splits)
            for split_idx in test_splits:
                #print(split_idx)
                pathCounter.update({'split'+str(split_idx):pathCounter['split'+str(split_idx)]+1})
                
            #print(pathCounter)
            
            train, tests = self.split_data(test_splits)
            trainSet = self.df.loc[self.df.erano.isin(train), self.features + [self.target, 'era']]
            
            allTestEras = []
            for t in range(len(test_splits)):
                allTestEras.append(tests[t]) 
                
            #print(allTestEras, train)    
            testSetFull = self.df.loc[self.df.erano.isin(allTestEras), self.features + [self.target, 'era']]
            
            self.model.fit(trainSet, testSetFull, com_idx)
            #print('fitted')
            
            for test, split_idx in zip(tests, test_splits):
                
                #print(test, split_idx)
                
                testSet = self.df.loc[self.df.erano.isin(test), self.features + [self.target, 'era']]
                
                split = 'split'+str(split_idx)
                path  = 'path' +str(pathCounter['split'+str(split_idx)])
                self.results.loc[split, path] = self.model.evaluate(testSet, neutralize)
            
            if save_file != None:
                self.results.to_csv('CV_results_'+save_file+'.csv')
                
            if save_model == True:
                joblib.dump(self.model.model, "CV_model_"+str(com_idx+run_from)+"-"+save_file+".model")
            
            gc.collect()
                
        return self.summary()

    def summary(self):
        
        corr = self.results.applymap(lambda x: json.loads(x)['corr']).mean().mean()
        std = self.results.applymap(lambda x: json.loads(x)['std']).mean().mean()
        sharpe = self.results.applymap(lambda x: json.loads(x)['sharpe']).mean().mean()
        drawdown = self.results.applymap(lambda x: json.loads(x)['drawdown']).min().min()
        lowest = self.results.applymap(lambda x: json.loads(x)['corr']).min().min()
        #sortino = self.results.applymap(lambda x: json.loads(x)['sortino']).min().min()

        print(corr, std, sharpe, drawdown, lowest)
        
        return corr, std, sharpe, drawdown#, sortino


    
class xgbModel():
    def __init__(self, features, target, **kwargs):
        
        self.features = features
        self.target = target
        self.model = xgboost. XGBRegressor(**kwargs)
        
    def fit(self, trainSet, a, b):
        
        self.model.fit(trainSet[self.features], trainSet[self.target])
          
    def evaluate(self, testSet, neutralize=0):
        
        testSet['pred'] = self.model.predict(testSet[self.features])
        
        if neutralize > 0:
            neutralize(testSet, neutralize, self.features)
        
        corr, std = numerai_score(testSet)
        max_drawdown = drawdown(testSet)
        
        res = {'corr':corr, 'std':std, 'sharpe':corr/std, 'drawdown':max_drawdown}
        return str(res).replace("'", '"')
    
#from cuml import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

class rfModel():
    def __init__(self, features, target, **kwargs):
        
        self.features = features
        self.target = target
        self.model = RandomForestRegressor(**kwargs)
        
    def fit(self, trainSet, a, b):
        
        self.model.fit(trainSet[self.features], trainSet[self.target])
          
    def evaluate(self, testSet, neutralize=0):
        
        testSet['pred'] = self.model.predict(testSet[self.features])
        
        if neutralize > 0:
            neutralize(testSet, neutralize, self.features)
        
        corr, std = numerai_score(testSet)
        max_drawdown = drawdown(testSet)
        
        res = {'corr':corr, 'std':std, 'sharpe':corr/std, 'drawdown':max_drawdown}
        return str(res).replace("'", '"')    
      