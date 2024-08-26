import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from octis.dataset.dataset import Dataset
from octis.models.GINOPIC import GINOPIC
from octis.evaluation_metrics.coherence_metrics import *
from octis.evaluation_metrics.diversity_metrics import *
from octis.evaluation_metrics.classification_metrics import *
import random, torch, json
from random import randint

data_dir = '../preprocessed_datasets'

def get_dataset(dataset_name):
    data = Dataset()
    if dataset_name=='20NG':
        data.fetch_dataset("20NewsGroup")
    elif dataset_name=='SO':
        data.load_custom_dataset_from_folder(data_dir + "/SO")
    elif dataset_name=='BBC':
        data.fetch_dataset("BBC_News")
    elif dataset_name=='Bio':
        data.load_custom_dataset_from_folder(data_dir + "/Bio")
    elif dataset_name=='SearchSnippets':
        data.load_custom_dataset_from_folder(data_dir + "/SearchSnippets")
    elif dataset_name=='BRNews':
        data.load_custom_dataset_from_folder(data_dir + "/BRNews")
    elif dataset_name=='hotels_reviews':
        data.load_custom_dataset_from_folder(data_dir + "/hotels_reviews")
    else:
        raise Exception('Missing Dataset name...!!!')
    return data

params = {
    '20NG': {
        'num_gin_layers': 2,
        'g_feat_size': 2048,
        'num_mlp_layers': 1,
        'gin_hidden_dim': 200,
        'gin_output_dim': 768,
        'eps_simGraph': 0.4
    },
    'BBC': {
        'num_gin_layers': 3,
        'g_feat_size': 256,
        'num_mlp_layers': 1,
        'gin_hidden_dim': 50,
        'gin_output_dim': 512,
        'eps_simGraph': 0.3
    },
    'Bio': {
        'num_gin_layers': 2,
        'g_feat_size': 1024,
        'num_mlp_layers': 1,
        'gin_hidden_dim': 200,
        'gin_output_dim': 256,
        'eps_simGraph': 0.05
    },
    'SO': {
        'num_gin_layers': 2,
        'g_feat_size': 64,
        'num_mlp_layers': 1,
        'gin_hidden_dim': 300,
        'gin_output_dim': 512,
        'eps_simGraph': 0.1
    },
    'SearchSnippets': {
        'num_gin_layers': 2,
        'g_feat_size': 1024,
        'num_mlp_layers': 1,
        'gin_hidden_dim': 50,
        'gin_output_dim': 256,
        'eps_simGraph': 0.2
    },
    'BRNews': {
        'num_gin_layers': 2,
        'g_feat_size': 1024,
        'num_mlp_layers': 1,
        'gin_hidden_dim': 50,
        'gin_output_dim': 256,
        'eps_simGraph': 0.4
    },
    'hotels_reviews': {
        'num_gin_layers': 2,
        'g_feat_size': 64,
        'num_mlp_layers': 1,
        'gin_hidden_dim': 50,
        'gin_output_dim': 256,
        'eps_simGraph': 0.2
    }
}

results = {
    'Dataset': [],
    'K': [],
    'Seed': [],
    'Model':[],
    'NPMI': [],
    'CV': [],
    'Accuracy': []
}

partition = True
validation = False

d = 'hotels_reviews'
seed = randint(0, 9999)
data = get_dataset(d)
k = 5

print("-"*100)
print('Dataset:{},\t K={},\t Seed={}'.format(d, k, seed))
print("-"*100)

random.seed(seed)
torch.random.manual_seed(seed)

model = GINOPIC(num_topics=k,
        use_partitions=partition,
        use_validation=validation,
        num_epochs=50,
        w2v_path='./w2v/{}_part{}_valid{}/'.format(d, partition, validation),
        graph_path='./doc_graphs/{}_part{}_valid{}/'.format(d, partition, validation),
        num_gin_layers=params[d]['num_gin_layers'],
        g_feat_size=params[d]['g_feat_size'],
        num_mlp_layers=params[d]['num_mlp_layers'],
        gin_hidden_dim=params[d]['gin_hidden_dim'],
        gin_output_dim=params[d]['gin_output_dim'],
        eps_simGraph=params[d]['eps_simGraph']
    )

output = model.train_model(dataset=data)