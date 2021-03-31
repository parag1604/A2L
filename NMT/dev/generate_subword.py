import random
import numpy as np
import time
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf

start_time = time.time()
base_path = "../global_data/Dataset/"


fp = open(base_path+"Europarl/europarl-v7.es-en.en",'r')
english_europarl = fp.readlines()
print(len(english_europarl))
fp.close()

fp = open(base_path+"Europarl/europarl-v7.es-en.es",'r')
spanish_europarl = fp.readlines()
print(len(spanish_europarl))
print(len(spanish_europarl))

total_datapoints = len(spanish_europarl)
total_corpus = []
for i in range(total_datapoints):
    total_corpus.append((english_europarl[i].strip(),spanish_europarl[i].strip()))

total_corpus = list(set(total_corpus))
total_datapoints = len(total_corpus)
print("Total datapoints are: ", total_datapoints)
print(total_corpus[444])

print("--- %s seconds ---" % (time.time() - start_time))


VOCAB_SIZE = 2**14
MAX_LENGTH = 100
tokenizer_src = pickle.load(
    open(base_path+'../Tokenizer/tok_en_' + str(VOCAB_SIZE) + '.pickle', 'rb'))
tokenizer_tar = pickle.load(
    open(base_path+'../Tokenizer/tok_pt_' + str(VOCAB_SIZE) + '.pickle', 'rb'))

print("Started")
start_time = time.time()
final_data = []
for i in total_corpus:
    en=tokenizer_src.encode(i[0])
    es=tokenizer_tar.encode(i[1])
    if len(en)>(MAX_LENGTH-2) or len(es)>(MAX_LENGTH-2):
        continue
    final_data.append(i)
print("total datapoints are:",len(final_data))
print("--- %s seconds ---" % (time.time() - start_time))


random.seed(10)
random.shuffle(final_data)
Labelled_corpus = final_data[0:200000]
Unlabelled_corpus =  final_data[200000:1000000]
print("Labelled: ",len(Labelled_corpus))
print("Unlabelled: ", len(Unlabelled_corpus))

with open(base_path+"../training_data/"+str(VOCAB_SIZE)+"/"+str(MAX_LENGTH)+"/Labelled.pickle", 'wb') as handle:
    pickle.dump(Labelled_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(base_path+"../training_data/"+str(VOCAB_SIZE)+"/"+str(MAX_LENGTH)+"/Unlabelled.pickle", 'wb') as handle:
    pickle.dump(Unlabelled_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Pickle Saved")
