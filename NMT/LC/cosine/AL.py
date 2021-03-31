import os
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


print("Cuda visible device ",sys.argv[4])
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[4]
global_data_path = "../../global_data/"


import tensorflow_datasets as tfds
import tensorflow as tf
from LSTM import *


import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import collections
import time
import random
import sacrebleu
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional




print("TF version is:",tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



Siamese_Model = []
VOCAB_SIZE = 2**13
MAX_LENGTH = 50
tokenizer_src = []
tokenizer_tar = []
loss_object=[]
optimizer = []
learning_rate=[]
embedding_dim = 256
units = 1024
encoder=""
decoder=""



def initialise_GV():
    global tokenizer_src, tokenizer_tar,encoder,decoder
    
    tokenizer_base_dir = global_data_path+"Tokenizer/"
    tokenizer_en_dir = tokenizer_base_dir+'tok_en_' + str(VOCAB_SIZE) + '.pickle'
    tokenizer_es_dir = tokenizer_base_dir+'tok_pt_' + str(VOCAB_SIZE) + '.pickle'
    
    
    print("Loading Tokenizer")
    tokenizer_src = pickle.load(open(tokenizer_en_dir, 'rb'))
    tokenizer_tar = pickle.load(open(tokenizer_es_dir, 'rb'))
    print("Total Vocab size in Source", tokenizer_src.vocab_size)
    print("Total Vocab size in Target", tokenizer_tar.vocab_size)
    
    vocab_inp_size = tokenizer_src.vocab_size + 2
    vocab_tar_size = tokenizer_tar.vocab_size + 2
    encoder = Encoder(vocab_inp_size, embedding_dim, units, 100)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, 100)
    
    

    
######################################## AL Strategies #########################

def get_max_len_for_pad(inp_sentence):
    temp_max_len=0
    for i in inp_sentence:
        if len(i)>temp_max_len:
            temp_max_len=len(i)
    return temp_max_len           


def add_post_pad(inp_sentence,testing):
    temp_max_len = 0
    if testing:
        temp_max_len = get_max_len_for_pad(inp_sentence)
    else:
        temp_max_len = MAX_LENGTH
    for i in range(len(inp_sentence)):
        while len(inp_sentence[i])<temp_max_len:
            inp_sentence[i].append(0)
        while len(inp_sentence[i])!=temp_max_len:
            inp_sentence[i].pop()
    return inp_sentence





def evaluate_multiplesentence(inp_sentence,testing, calc_prob=False,cal_attention=False):
    no_of_inp_sentence = len(inp_sentence)
    start_token = [tokenizer_src.vocab_size]
    end_token = [tokenizer_src.vocab_size + 1]

    if testing:
        inp_sentence = [tokenizer_src.encode(x) for x in inp_sentence]
    else: 
        inp_sentence = [tokenizer_src.encode(x[0]) for x in inp_sentence]

    inp_sentence = [start_token + x + end_token for x in inp_sentence ]  
    
    
    source_endlist = []
    if cal_attention:
        for i in inp_sentence:
            if len(i)<=MAX_LENGTH:
                source_endlist.append(len(i))
            else:
                source_endlist.append(MAX_LENGTH)
    
    
    inp_sentence = add_post_pad(inp_sentence,testing)  

    encoder_input = tf.convert_to_tensor(inp_sentence)
    dec_input = [[tokenizer_tar.vocab_size] for i in range(no_of_inp_sentence)]
    output = tf.convert_to_tensor(dec_input)
    dec_input = tf.convert_to_tensor(dec_input)

    
    hidden = (tf.zeros((no_of_inp_sentence, units)),tf.zeros((no_of_inp_sentence, units)))
    enc_out, enc_hidden = encoder(encoder_input, hidden)
    
    enc_mask = (tf.cast(tf.math.equal(encoder_input, 0), tf.float32) * -1e9) 
    enc_mask = tf.expand_dims(enc_mask, -1)
    
    dec_hidden = enc_hidden

    
    attention_weigths_np = ""
    if cal_attention:
        attention_weigths_np = np.zeros((MAX_LENGTH,no_of_inp_sentence,MAX_LENGTH))
    
    
    output_prob = ""
    if calc_prob:
        output_prob = tf.cast(tf.convert_to_tensor([[1] for i in range(no_of_inp_sentence)]),tf.float32) 
    
    for t in range(MAX_LENGTH):
        predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out,enc_mask)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        dec_input = tf.reshape(predicted_id, (predicted_id.shape[0],-1))
        
        output = tf.concat([output, dec_input], axis=-1)
        if calc_prob:
            predicted_prob = tf.nn.softmax(predictions)
            predicted_prob = tf.reshape(tf.reduce_max(predicted_prob,axis=-1), (-1,1))
            output_prob = tf.concat([output_prob, predicted_prob], axis=-1)
         
        if cal_attention:
            attention_weigths_np[t] = attention_weights.numpy().reshape(-1,MAX_LENGTH)
        
    return output,attention_weigths_np,output_prob,source_endlist

def make_partitions(data_list,batch_size):
    partitions = []
    for batch_num in range(int(np.ceil(len(data_list)/batch_size))):
        start_index = batch_num*batch_size
        end_index = start_index + batch_size
        if end_index>len(data_list):
            end_index = len(data_list)
        partitions.append(data_list[start_index:end_index])
    return partitions

def get_target_endlist(output):
    target_endlist = []
    for sentence in output:
        temp = 0
        for token in sentence:
            if(token==tokenizer_tar.vocab_size+1):
                break
            else:
                temp = temp+1
        if temp>MAX_LENGTH:
            target_endlist.append(MAX_LENGTH)
        else:
            target_endlist.append(temp)
    return target_endlist



# def ADS(attention_matrix,len_x,len_y):
#     kurt = []
#     for i in range(len_y):
#         num_sum = 0
#         den_sum = 0
#         for j in range(len_x):
#             alpha = attention_matrix[i,j]
#             num_sum += np.power((alpha - 1/len_x),4)
#             den_sum += np.power((alpha - 1/len_x),2)
#         num_sum = num_sum/len_x
#         den_sum = np.power((den_sum/len_x),2)
#         kurt.append(num_sum/den_sum)

#     ADS_score = (-sum(kurt)/len_y)
#     return ADS_score
    
    
from scipy.stats import kurtosis
def get_ADS_Scores(chosen_unlabelled_data,eval_batch_size):
    print("Evaluating data and getting ADS scores")
    
    start_time = time.time()
    partions = make_partitions(chosen_unlabelled_data,eval_batch_size)
    
    ADS_scores = []
    
    for batch_num,to_translate in enumerate(partions):
        
        print(batch_num)
        retval_eval=[]
        with tf.device('/device:GPU:0'):
            retval_eval = evaluate_multiplesentence(to_translate,False,False,True)
        
        target_endlist = get_target_endlist(retval_eval[0])
        attention_np = retval_eval[1].transpose((1,0,2))
        source_endlist = retval_eval[3]
        
        for point in range(len(to_translate)):
            len_y = target_endlist[point]
            len_x = source_endlist[point] 
            
            alphas = attention_np[point,0:len_y,0:len_x]        
            lib_score = -kurtosis(alphas.T).mean()#use fisher=False to get similar value to the fucntion       
            ADS_scores.append(lib_score)
    print("--- %s seconds ---" % (time.time() - start_time)) #50sec for 512 in one gpu
    return ADS_scores



def get_CS_Scores(chosen_unlabelled_data,eval_batch_size):
    print("Evaluating data and getting CS scores")
    
    start_time = time.time()
    partions = make_partitions(chosen_unlabelled_data,eval_batch_size)
    
    CS_scores = []
    for batch_num,to_translate in enumerate(partions):
        
        print(batch_num)
        retval_eval=[]
        with tf.device('/device:GPU:0'):
            retval_eval = evaluate_multiplesentence(to_translate,False,False,True)
        
        target_endlist = get_target_endlist(retval_eval[0])
        attention_np = retval_eval[1].transpose((1,0,2))
        source_endlist = retval_eval[3]
        
        for point in range(len(to_translate)):
            len_y = target_endlist[point]
            len_x = source_endlist[point] 
            alphas = attention_np[point,0:len_y,0:len_x]     
            
            cp_penalty = 0.0
            for cp_j in range(len_x):
                att_weight = 0.0
                for cp_i in range(len_y):
                    att_weight += alphas[cp_i, cp_j]
                cp_penalty += np.log(min(att_weight, 1.0))
            
            score = float(cp_penalty) / len_x
            score = -score
            CS_scores.append(score)
    print("--- %s seconds ---" % (time.time() - start_time)) #50sec for 512 in one gpu
    return CS_scores



def get_predicted_probabilities(chosen_unlabelled_data,eval_batch_size):
    print("Evaluating data and getting probabilities")
    start_time = time.time()
    predicted_probabilities = []
    partions = make_partitions(chosen_unlabelled_data,eval_batch_size)
    for batch_num,to_translate in enumerate(partions):
        print(batch_num)
        retval_eval=[]
        with tf.device('/device:GPU:0'):
            retval_eval = evaluate_multiplesentence(to_translate,False,True)
        batch_predicted_translations = retval_eval[0].numpy()
        batch_predicted_probabilities = retval_eval[2].numpy()
        for j in range(0,len(to_translate)):
            prob_list=[]
            for num,i in enumerate(batch_predicted_translations[j]):
                if (i < tokenizer_tar.vocab_size and i>0):
                    prob_list.append(batch_predicted_probabilities[j][num])
                if(i==tokenizer_tar.vocab_size+1):
                    prob_list.append(batch_predicted_probabilities[j][num])
                    break
            predicted_probabilities.append(prob_list)
    print("--- %s seconds ---" % (time.time() - start_time)) #50sec for 512 in one gpu
    return predicted_probabilities



def get_LC_Scores(chosen_unlabelled_data,eval_batch_size):
    print("Using Least_Confidence Strategy")
    predicted_probabilities = get_predicted_probabilities(chosen_unlabelled_data,eval_batch_size)
    scores = []
    for i in predicted_probabilities:
        if len(i)==0:
            scores.append(-1)
            continue
        temp = 0
        for j in i:
            temp = temp+np.log(j)
        temp = -temp/len(i)
        scores.append(temp)
    print("Done")
    return scores
##########################################################################################
    

    

    




########################## Cosine Similarity #######################
def get_encoder_embeddings(siamese_input):
    enc_output,state = encoder(siamese_input,None)#,hidden)
    return state[0]



def siamese_get_max_len_for_pad(inp_sentence):
    temp_max_len=0
    for i in inp_sentence:
        if len(i)>temp_max_len:
            temp_max_len=len(i)
    return temp_max_len


def siamese_add_post_pad(inp_sentence):
    temp_max_len = siamese_get_max_len_for_pad(inp_sentence)
    for i in range(len(inp_sentence)):
        while len(inp_sentence[i])<temp_max_len:
            inp_sentence[i].append(0)
    return inp_sentence


def get_siamese_embeddings(inp_sentence):
  no_of_inp_sentence = len(inp_sentence)
  start_token = [tokenizer_src.vocab_size]
  end_token = [tokenizer_src.vocab_size + 1]
  inp_sentence = [tokenizer_src.encode(x[0]) for x in inp_sentence]
  inp_sentence = [start_token + x + end_token for x in inp_sentence ]
  inp_sentence = siamese_add_post_pad(inp_sentence)  
  siamese_input = tf.convert_to_tensor(inp_sentence)

  #cosine similarity
  fun_out = get_encoder_embeddings(siamese_input)

#   print(fun_out.shape)
  return fun_out.numpy().tolist()



def siamese_embeddings_util(AL_samples,eval_batch_size):
    print("Getting encoder hidden embeddings")
    siamese_embedlist = []
    for batch_num in range( int(np.ceil(len(AL_samples)/eval_batch_size)) ):
        start_num = batch_num*eval_batch_size
        end_num = start_num + eval_batch_size
        if end_num>len(AL_samples):
            end_num = len(AL_samples)
        to_embed = AL_samples[start_num:end_num]
        with tf.device('/device:GPU:0'):
            batch_predicions = get_siamese_embeddings(to_embed)
            siamese_embedlist =  siamese_embedlist+batch_predicions
    return siamese_embedlist



def get_representative_samples(siamese_embeddings,no_of_samples,samples_per_cluster,no_of_clusters):
    
    #Cosine Similarity
    similarity_matrix = cosine_similarity(siamese_embeddings, siamese_embeddings)
    similarity_matrix = np.exp(similarity_matrix)
    
    
    #returns array of length AL_NUM where each element tells which cluster that index belong to
    clustering = SpectralClustering(n_clusters=no_of_clusters, random_state=0, affinity='precomputed').fit(similarity_matrix)
    clusters = collections.defaultdict(list)
    for idx, label in enumerate(clustering.labels_):
        clusters[label].append(idx) 
    
    representative_samples = []
    total_samples = [i for i in range(0, siamese_embeddings.shape[0])]
    for i in clusters:
        if len(clusters[i])>samples_per_cluster:
            representative_samples +=clusters[i][0:samples_per_cluster]
        else:
            representative_samples +=clusters[i]
    remaining_samples = list(set(total_samples).difference(set(representative_samples)))
    representative_samples += remaining_samples[ 0: (no_of_samples-len(representative_samples))  ]
    return representative_samples


def A2L(AL_samples,no_of_samples,samples_per_cluster,no_of_clusters,eval_batch_size):
    print("Cosine Similarity")
    siamese_embeddings = np.array(siamese_embeddings_util(AL_samples,eval_batch_size))
    representative_samples = get_representative_samples(siamese_embeddings,no_of_samples,samples_per_cluster,no_of_clusters)
    A2L_samples = [AL_samples[i] for i in representative_samples]
    print("Done")
    return A2L_samples
####################################################





########################################### Build Model #########################################
def build_model():
    print("building the model")
    inputs = tf.convert_to_tensor(  [[1,0]]  )
    
    hidden = (tf.zeros((1, units)),tf.zeros((1, units)))
    enc_out, enc_hidden = encoder(inputs, hidden)
    enc_mask = (tf.cast(tf.math.equal(inputs, 0), tf.float32) * -1e9) 
    enc_mask = tf.expand_dims(enc_mask, -1)
    
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([0], 0)
    predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out,enc_mask)
    print("model built")
#################################################################################################
    
    
    
def main():
    
    
    global MAX_LENGTH,VOCAB_SIZE
    A2L_Flag = int(sys.argv[1])
    MAX_LENGTH = int(sys.argv[2])
    VOCAB_SIZE = int(sys.argv[3])
    
    
    chosen_unlabelled_data = []
    with open("./sample_data.pickle", 'rb') as handle:
        chosen_unlabelled_data = pickle.load(handle)
    
    SAMPLE_NUM = len(chosen_unlabelled_data)
    AL_NUM =  int(SAMPLE_NUM/2)
    A2L_NUM = int(SAMPLE_NUM/10)
    no_of_clusters = 50
    per_cluster_samples = A2L_NUM//no_of_clusters
   

    AL_Strategy = "LC"
    
    
    initialise_GV()
    build_model()
    print("Loading Model weights")
    encoder.load_weights('./LSTM_ckpt/encoder.h5')
    decoder.load_weights('./LSTM_ckpt/decoder.h5')
    print("Done")
 

    eval_batch_size = 1150    
    print("Batch size is:-",eval_batch_size)
    

    chosen_samples = []
    Scores = ""
    if AL_Strategy == "ADS":
        print("AL_Strategy:- Attention distraction sampling")
        Scores = get_ADS_Scores(chosen_unlabelled_data,eval_batch_size)
        
    if AL_Strategy == "LC":
        print("AL_Strategy:- Least Confidence")
        Scores = get_LC_Scores(chosen_unlabelled_data,eval_batch_size)
    
    if AL_Strategy=="CS":
        print("AL_Strategy:- Coverage Sampling")
        Scores = get_CS_Scores(chosen_unlabelled_data,eval_batch_size)
    
    Scores = np.array(Scores)
    sorted_indices = (-Scores).argsort()[:AL_NUM]
    AL_samples = [chosen_unlabelled_data[i]  for i in sorted_indices]
    chosen_samples = AL_samples
    
    

    
    print("Cosine Similarity")
    start_time = time.time()
    chosen_samples = A2L(chosen_samples,A2L_NUM,per_cluster_samples,no_of_clusters,eval_batch_size)
    print("--- %s seconds ---" % (time.time() - start_time))
        
    with open("./chosen_data.pickle", 'wb') as handle:
        pickle.dump(chosen_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__": 
    main()
    