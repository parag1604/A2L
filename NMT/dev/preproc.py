import os, sys
import tensorflow as tf
import tensorflow_datasets as tfds
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import re, pickle
from nltk.tokenize import RegexpTokenizer
tokenize = RegexpTokenizer("[\w']+").tokenize



global_data_path = "../global_data/"

VOCAB_SIZE = 2**13
MAX_LENGTH = 50

training_data_dir = global_data_path+"training_data/"
labelled_dir = training_data_dir+str(VOCAB_SIZE)+"/"+str(MAX_LENGTH)+"/Labelled.pickle"
unlabelled_dir = training_data_dir+str(VOCAB_SIZE)+"/"+str(MAX_LENGTH)+"/Unlabelled.pickle"

data = []
with open(labelled_dir, 'rb') as handle:
    data.extend(pickle.load(handle))
with open(unlabelled_dir, 'rb') as handle:
    data.extend(pickle.load(handle))
# data = data[:5]
data2 = []
eng, esp = [], []
for datum in data:
	en_str = ' '.join(tokenize(re.sub(r"[' ]+[ ']", " ", datum[0].lower())))
	es_str = ' '.join(tokenize(re.sub(r"[' ]+[ ']", " ", datum[1].lower())))
	data2.append((en_str, es_str))
	eng.append(en_str)
	esp.append(es_str)
data = data2

with open(global_data_path+"word_level/Labelled.pickle", 'wb') as handle:
    pickle.dump(data[:200000], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(global_data_path+"word_level/Unlbelled.pickle", 'wb') as handle:
    pickle.dump(data[200000:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(global_data_path+"word_level/ENG.pickle", 'wb') as handle:
    pickle.dump(eng, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(global_data_path+"word_level/ESP.pickle", 'wb') as handle:
    pickle.dump(esp, handle, protocol=pickle.HIGHEST_PROTOCOL)
