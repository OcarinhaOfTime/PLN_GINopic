import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from octis.dataset.dataset import Dataset
import random, torch, json

data = Dataset()
data_dir = '../preprocessed_datasets'
data.load_custom_dataset_from_folder(data_dir + "/BRNews")

meta = data.get_metadata()
print(meta)
print(meta["last-training-doc"])
#print("last-training-doc" in meta)
pcorpus = data.get_partitioned_corpus()
print(pcorpus)