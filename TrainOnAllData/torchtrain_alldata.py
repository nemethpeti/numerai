import torch
import pandas as pd
import numpy as np
import torchsort
from torch.functional import F
import torch.optim as optim
from torch import nn

import json
import joblib
import random
import math


#device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

torch.use_deterministic_algorithms(mode=True)
pd.options.mode.chained_assignment = None  # default='warn'

import numeraiutilsv2 as num
from numeraiutilsv2 import xgbModel

# In[ data defaults ]

mnt = 'C://' # /mnt/c/
dataDir =  mnt+'Users/Newton/Documents/Trading/Numerai/numerai_datasets'
modelDir = mnt+'Users/Newton/Documents/Trading/Numerai/autosubmit/trained_models_v2'

keys = ['era', 'data_type']

targets = ['target',
 'target_nomi_v4_20',
 'target_nomi_v4_60',
 'target_jerome_v4_20',
 'target_jerome_v4_60',
 'target_janet_v4_20',
 'target_janet_v4_60',
 'target_ben_v4_20',
 'target_ben_v4_60',
 'target_alan_v4_20',
 'target_alan_v4_60',
 'target_paul_v4_20',
 'target_paul_v4_60',
 'target_george_v4_20',
 'target_george_v4_60',
 'target_william_v4_20',
 'target_william_v4_60',
 'target_arthur_v4_20',
 'target_arthur_v4_60',
 'target_thomas_v4_20',
 'target_thomas_v4_60']

badFeatures = ['feature_palpebral_univalve_pennoncel',
 'feature_unsustaining_chewier_adnoun',
 'feature_brainish_nonabsorbent_assurance',
 'feature_coastal_edible_whang',
 'feature_disprovable_topmost_burrower',
 'feature_trisomic_hagiographic_fragrance',
 'feature_queenliest_childing_ritual',
 'feature_censorial_leachier_rickshaw',
 'feature_daylong_ecumenic_lucina',
 'feature_steric_coxcombic_relinquishment']


# select training set size
target = 'target'

f = open(dataDir+'/features.json')
j = json.load(f)

# select feature set
features = list(j['feature_stats'].keys())
#features = j['feature_sets']["medium"]

# remove bad features
features = [f for f in features if f not in badFeatures]
n_features = len(features)
        

# In[define dataset]


class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(n_features, n_features//2)
        self.lin2 = nn.Linear(n_features//2, n_features//4)
        self.lin3 = nn.Linear(n_features//4, 1)
              
        self.do1 = nn.Dropout(0.1)
        self.do2 = nn.Dropout(0.1)
        self.do3 = nn.Dropout(0.1)

    def forward(self, input):
        
        layer1 = self.do1(       F.relu(self.lin1(input)))
        layer2 = self.do2(       F.relu(self.lin2(layer1)))
        out    = self.do3(torch.sigmoid(self.lin3(layer2)))    # first output, predicting labels
        
        return out

def numerai_corr(pred, target, regularization_strength=.0001):
    # Computes and returns a Numerai score and feature exposure
    
    pred = pred.reshape(1, -1)
    target = target.reshape(1, -1)
    
    # get sorted indicies
    rr = torchsort.soft_rank(pred, regularization_strength=regularization_strength)
    # change pred to uniform distribution
    pred = (rr - .5)/rr.shape[1]
    
    # Pearson correlation
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    corr = (pred * target).sum()
    
    return corr


class NumeraiDataSet(torch.utils.data.IterableDataset):
    
    def __init__(self, splitDir, era_start, era_end, features, target):
        
        super(NumeraiDataSet).__init__()
        self.dir = splitDir
        
        assert era_start<era_end
        self.start = era_start
        self.end = era_end
        self.features = features
        self.target = target
        self.current = self.start

    def __iter__(self):
        
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process

            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)        
        
        #print(worker_id, ' - ', iter_start, iter_end)
        
        eras = list(range(iter_start, iter_end))
        random.shuffle(eras)
        
        for idx in eras:
            file = f'{self.dir}{idx:04d}.parquet'
            df = pd.read_parquet(file, columns=features+[target])
            
            # shuffle
            df = df.sample(frac=1).reset_index(drop=True)
            
            yield df[features].values-0.5, df[target].values, idx  
        #return file
        
if __name__=='__main__':
            
    learning_rate=0.001
    model=Net(n_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model.train()

    training = NumeraiDataSet(dataDir+'/eraSplit/', 1, 800, features, target)
    training_loader = torch.utils.data.DataLoader(training, num_workers=5, batch_size=1)

    validation = NumeraiDataSet(dataDir+'/eraSplit/', 800, 1028, features, target)
    validation_loader = torch.utils.data.DataLoader(validation, num_workers=5, batch_size=1)

    
    torch.autograd.set_detect_anomaly(True)
    regularization_strength=.0001
    epochs=6
      
    for epoch in range(epochs):
        print('Epoch', epoch)

        for ii, data in enumerate(training_loader):
            
            # get features and target from data and put in tensors
            X, y, idx = data
            X = torch.squeeze(X).to(device)
            y = torch.squeeze(y).to(device)
    
            # zero gradient buffer and get model output
            optimizer.zero_grad()
            model.train()
            out = model(X)
            
            corr = numerai_corr(out, y)
            
            loss = -corr
    
            loss.backward()
            optimizer.step()
            
            if (ii)%100==0:
                print(ii, corr.item())
                
            
        # end of epoch validation
        val = pd.DataFrame(columns=['era', 'target', 'pred'], dtype=[str, float, float])
        model.eval()
        with torch.no_grad():
            
            for ii, data in enumerate(validation_loader):
                X, y, idx = data
                X = torch.squeeze(X).to(device)
                y = y.cpu().detach().numpy()
            
                out = model(X)
                y_hat = out.cpu().detach().numpy()
                
                era = {'target':y.squeeze(), 'pred':y_hat.squeeze()}
                era = pd.DataFrame(era)
                era['era'] = f'{idx.cpu().detach().numpy()[0]:04d}'
                
                val = pd.concat([val, era])
                
            num.fast_evaluation(val, features)
        
        torch.save(model, f'epoch-{epoch}.pth')
        model.train()
        
