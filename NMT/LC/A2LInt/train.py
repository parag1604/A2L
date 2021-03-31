import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


global_data_path = "../../global_data/"
print("Cuda visible device ",sys.argv[5])
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[5]




import tensorflow_datasets as tfds
import tensorflow as tf
from LSTM import *


import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.cluster import SpectralClustering
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


# Global Variables
BUFFER_SIZE = 20000
BATCH_PER_REPLICA = 125
BATCH_SIZE = BATCH_PER_REPLICA * strategy.num_replicas_in_sync



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







def load_optimizers():
    global optimizer,loss_object
    print("Loading optimizers")
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')



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
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_PER_REPLICA)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_PER_REPLICA)



def add_pad(lst):
    while len(lst)<MAX_LENGTH:
        lst.append(0)
    return lst


def get_traindataset(train_data):




    print("Loading Dataset")
    random_seed = 7


    start_time = time.time()


    final_en = []
    final_es = []
    start_time = time.time()
    en_toklist = [[tokenizer_src.vocab_size] + tokenizer_src.encode(i[0])+ [tokenizer_src.vocab_size + 1] for i in train_data]
    es_toklist = [[tokenizer_tar.vocab_size] + tokenizer_tar.encode(i[1])+ [tokenizer_tar.vocab_size + 1] for i in train_data]

    for i in range(len(en_toklist)):
        if len(en_toklist[i])>MAX_LENGTH or len(es_toklist[i])>MAX_LENGTH:
            continue
        final_en.append(  add_pad(en_toklist[i])   )
        final_es.append(  add_pad(es_toklist[i])   )

    final_en = np.array(final_en,dtype=np.int64)
    final_es = np.array(final_es,dtype=np.int64)

    train_dataset = tf.data.Dataset.from_tensor_slices((final_en,final_es))

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=random_seed,reshuffle_each_iteration=True).batch(BATCH_SIZE,drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    total_elem = tf.data.experimental.cardinality(train_dataset).numpy()

    print("--- %s seconds for making the dataset ---" % (time.time() - start_time)) #32sec
    print("Training Started")
    print("Total datapoints:-", total_elem*BATCH_SIZE)
    print("Total batches:-", total_elem)
    return train_dataset



def loss_function(real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = loss_object(real, pred)
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
      return tf.reduce_mean(loss_)
#       return tf.nn.compute_average_loss(loss_, global_batch_size=BATCH_SIZE)



@tf.function()
def train_step(inp, targ):
    loss = 0
    enc_hidden = encoder.initialize_hidden_state()
    enc_mask = (tf.cast(tf.math.equal(inp, 0), tf.float32) * -1e9)
    enc_mask = tf.expand_dims(enc_mask, -1)


    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(targ[:, 0], 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output,enc_mask)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)


    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss



def train(train_data,EPOCHS):
    train_dataset = get_traindataset(train_data)
    steps_per_epoch = len(train_data)//BATCH_SIZE
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset):
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss
            if batch==0 and epoch==0:
                print('Time taken for building graph is {} sec\n'.format(time.time() - start))
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
##############################################################################################################





############################ Loading and finding bleu score of test data #####################################
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


def get_test_bleu(en_toklist_test,es_toklist_test,eval_batch_size):
    print("Evaluating Test data")
    start_time = time.time()
    predicted_translations = []
    partions = make_partitions(en_toklist_test,eval_batch_size)
    for batch_num,to_translate in enumerate(partions):
        print(batch_num)
        batch_predicted_translations = []
        with tf.device('/device:GPU:0'):
            batch_predicted_translations = evaluate_multiplesentence(to_translate,True)[0].numpy()
        for j in range(0,len(to_translate)):
            temp_list = []
            for i in batch_predicted_translations[j]:
                if (i < tokenizer_tar.vocab_size and i>0):
                    temp_list.append(i)
                if(i==tokenizer_tar.vocab_size+1):
                    break
            predicted_sentence = tokenizer_tar.decode(temp_list)
            predicted_translations.append(predicted_sentence)

    print("--- %s seconds for testing---" % (time.time() - start_time)) #50sec for 512 in one gpu
    refs = [es_toklist_test]
    bleu = sacrebleu.corpus_bleu(predicted_translations, refs)
    return bleu.score





def load_testdata():
    print("Loading Test data")


    en_test_path = global_data_path+"test_data/newstest2013.en"
    en_toklist_test = open(en_test_path,"r").readlines()
    es_test_path = global_data_path+"test_data/newstest2013.es"
    es_toklist_test = open(es_test_path,"r").readlines()

    for i in range(0,len(es_toklist_test)):
        es_toklist_test[i] = es_toklist_test[i].strip()
        en_toklist_test[i] = en_toklist_test[i].strip()
    return en_toklist_test,es_toklist_test
##############################################################################################################






SICK_DATA = []
SIAMESE_MAX_LENGTH = 75
SIAMESE_BUFFER_SIZE = 20000
SIAMESE_BATCH_SIZE = 128
Siamese_Model = []
siamese_optimizer = []
siamese_loss = []
int_cluster_model = []
########################### Loading Sick Dataset ##########################################################
def sick_encode(lang1, lang2):
    lang1 = [tokenizer_src.vocab_size] + tokenizer_src.encode(
        lang1.numpy()) + [tokenizer_src.vocab_size + 1]

    lang2 = [tokenizer_src.vocab_size] + tokenizer_src.encode(
        lang2.numpy()) + [tokenizer_src.vocab_size + 1]
    return lang1, lang2

def sick_encode2(lang1, lang2, lang3):
    lang1 = [tokenizer_src.vocab_size] + tokenizer_src.encode(
        lang1.numpy()) + [tokenizer_src.vocab_size + 1]

    lang2 = [tokenizer_src.vocab_size] + tokenizer_src.encode(
        lang2.numpy()) + [tokenizer_src.vocab_size + 1]

    lang3 = [tokenizer_src.vocab_size] + tokenizer_src.encode(
        lang3.numpy()) + [tokenizer_src.vocab_size + 1]

    return lang1, lang2, lang3


def sick_tf_encode(src, tar, lab):
    result_src, result_tar = tf.py_function(sick_encode, [src, tar], [tf.int64, tf.int64])
    result_src.set_shape([None])
    result_tar.set_shape([None])
    return result_src, result_tar,lab

def sick_tf_encode2(src, pos, neg):
    result_src, result_pos, result_neg = tf.py_function(sick_encode2,
                                                        [src, pos, neg],
                                                        [tf.int64, tf.int64, tf.int64])
    result_src.set_shape([None])
    result_pos.set_shape([None])
    result_neg.set_shape([None])
    return result_src, result_pos, result_neg

def sick_filter_max_length(x, y, z, max_length=SIAMESE_MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)

def sick_filter_max_length2(x, y, z, max_length=SIAMESE_MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.logical_and(tf.size(y) <= max_length,
                                         tf.size(z) <= max_length))

def sick_generate_dataset():
    tot_len = len(SICK_DATA)
    for i in range(tot_len):
        yield (SICK_DATA[i][0], SICK_DATA[i][1] ,  SICK_DATA[i][2])

def sick_generate_dataset2():
    tot_len = len(SICK_DATA)
    for i in range(tot_len):
        j = i
        while j != i:
            j = np.random.choice(len(SICK_DATA), 2)[0]
        yield (SICK_DATA[i][0], SICK_DATA[i][1] ,  SICK_DATA[j][1])

def get_siamese_data():
    global SICK_DATA
    sick_path = global_data_path+"Sick/SICK.txt"
    sick_lines = open(sick_path,'r').readlines()
    for i,a in enumerate(sick_lines):
        if i==0:
            continue
        b = a.split("\t")
        SICK_DATA.append( (b[1],b[2], float(b[4])/5   )   )

    sick_examples = tf.data.Dataset.from_generator(sick_generate_dataset, (tf.string, tf.string,tf.float32), ((), (),()))
    sick_dataset = sick_examples.map(sick_tf_encode)
    sick_dataset = sick_dataset.filter(sick_filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    sick_dataset = sick_dataset.cache()
    sick_dataset = sick_dataset.shuffle(SIAMESE_BUFFER_SIZE,seed=7).padded_batch(SIAMESE_BATCH_SIZE, padded_shapes=((None,), (None,), ()) )
    sick_dataset = sick_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return sick_dataset

def get_int_cluster_data():
    global SICK_DATA

    sick_examples = tf.data.Dataset.from_generator(sick_generate_dataset2, (tf.string, tf.string, tf.string), ((), (),()))
    sick_dataset = sick_examples.map(sick_tf_encode2)
    sick_dataset = sick_dataset.filter(sick_filter_max_length2)
    # cache the dataset to memory to get a speedup while reading from it.
    sick_dataset = sick_dataset.cache()
    sick_dataset = sick_dataset.shuffle(SIAMESE_BUFFER_SIZE,seed=7).padded_batch(SIAMESE_BATCH_SIZE, padded_shapes=((None,), (None,), (None,)) )
    sick_dataset = sick_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return sick_dataset

#############################################################################################################





########################### Training Siamese Model ##########################################################

def get_enc_output(encoder_input):
    enc_output,state = encoder(encoder_input,None)#,hidden)
    return enc_output


class Siamese(tf.keras.Model):
    def __init__(self, dropout,recurrent_dropout, input_dim, hidden_dim):
        super(Siamese, self).__init__()
        self.bidir = Bidirectional(LSTM(hidden_dim, return_sequences=False,dropout=dropout,
                                        recurrent_dropout=recurrent_dropout), input_shape=(None, input_dim) )
    def call(self, inp1,inp2):
        transformer_embed1 = get_enc_output(inp1)
        transformer_embed2 = get_enc_output(inp2)
        mask1 = tf.cast(tf.math.not_equal(inp1, 0), tf.bool)
        mask2 = tf.cast(tf.math.not_equal(inp2, 0), tf.bool)
        bidir_output1 = self.bidir(transformer_embed1,training=True,mask=mask1)
        bidir_output2 = self.bidir(transformer_embed2,training=True,mask=mask2)
        return tf.math.exp(-tf.norm((bidir_output1-bidir_output2),axis=-1,ord=2))

    def get_lstm_output(self,inp1):
        transformer_embed1 = get_enc_output(inp1)
        mask = tf.cast(tf.math.not_equal(inp1, 0), tf.bool)
        bidir_output1 = self.bidir(transformer_embed1,training=False,mask=mask)
        return bidir_output1


def siamese_loss_function(real, pred):
        abs_loss = tf.abs(real-pred)
    #     return tf.reduce_sum(abs_loss)
        return tf.reduce_mean(abs_loss)/2





siamese_train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None,),dtype=tf.float32)
]
@tf.function(input_signature=siamese_train_step_signature)
def siamese_train_step(inp1, inp2, lab):
    with tf.GradientTape() as tape:
        similarity = Siamese_Model(inp1,inp2)
        loss = siamese_loss_function(lab, similarity)
    gradients = tape.gradient(loss, Siamese_Model.trainable_variables)
    siamese_optimizer.apply_gradients(zip(gradients, Siamese_Model.trainable_variables))
    siamese_loss(loss)




def train_siamese(sick_dataset,SIAMESE_EPOCHS):
    print("Siamese Training Started")
    with tf.device('/device:GPU:0'):
        for epoch in range(SIAMESE_EPOCHS):
            start = time.time()
            siamese_loss.reset_states()
            for (batch, (inp1,inp2,lab)) in enumerate(sick_dataset):
                siamese_train_step(inp1,inp2,lab)
                if batch % 40 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, siamese_loss.result()))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, siamese_loss.result(),))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print("Siamese training finished")



def load_siamese_optimizers():
    global siamese_optimizer,siamese_loss,Siamese_Model
    print("Loading Siamese optimizer")
    siamese_optimizer = tf.keras.optimizers.Adam() #previously 10^-5
    siamese_loss = tf.keras.metrics.Mean(name='train_loss')


def create_siamese_model():
    global Siamese_Model
    print("Creating Siamese Model")
    Siamese_Model = Siamese(0,0,units,300)
    load_siamese_optimizers()
    print("Done")
#############################################################################################################





############################ INTEGRATED CLUSTERING MODEL ######################################

class IntCluster(tf.keras.Model):

    def __init__(self):
        super(IntCluster, self).__init__()
        xavier=tf.keras.initializers.GlorotUniform()
        self.fc=tf.keras.layers.Dense(256,
                                      kernel_initializer=xavier,
                                      activation=tf.nn.relu,
                                      input_shape=[600,])
        self.out=tf.keras.layers.Dense(50,
                                       kernel_initializer=xavier)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=1e-6)

    # Running the model
    def run(self, X):
        global Siamese_Model
        X = Siamese_Model.get_lstm_output(X)
        return tf.math.argmax(tf.nn.softmax(self.out(self.fc(X))), axis=1)

    #Custom loss function
    def get_loss(self, X, Y, Z, Lambda):
        global Siamese_Model
        X = Siamese_Model.get_lstm_output(X)
        Y = Siamese_Model.get_lstm_output(Y)
        Z = Siamese_Model.get_lstm_output(Z)
        prob_a=tf.nn.softmax(self.out(self.fc(X)))
        prob_b=tf.nn.softmax(self.out(self.fc(Y)))
        prob_c=tf.nn.softmax(self.out(self.fc(Z)))
        idxs = np.zeros(prob_a.shape, dtype=np.bool)
        for i, j in enumerate(tf.math.argmax(prob_a, axis=1)):
            idxs[i, j] = True
        loss1 = -1 * Lambda[0] * tf.math.log(prob_b[idxs])
        loss2 = -1 * Lambda[1] * tf.math.log(1. - prob_c[idxs])
        log_prob_b = tf.math.log(prob_b)
        loss3 = Lambda[2] * tf.reduce_sum(prob_b*log_prob_b, axis=1)
        loss = tf.reduce_mean(loss1 + loss2 + loss3)
        # print(loss1)
        # print(loss2)
        # print(loss3)
        return loss

    # get gradients
    def get_grad(self, X, Y, Z, Lambda):
        with tf.GradientTape() as tape:
            tape.watch(self.fc.variables)
            tape.watch(self.out.variables)
            L = self.get_loss(X, Y, Z, Lambda)
            g = tape.gradient(L, [self.fc.variables[0],
                                  self.fc.variables[1],
                                  self.out.variables[0],
                                  self.out.variables[1]])
        return g, L

    # perform gradient descent
    def network_learn(self, X, Y, Z, Lambda):
        g, L = self.get_grad(X, Y, Z, Lambda)
        self.train_op.apply_gradients(zip(g, [self.fc.variables[0],
                                              self.fc.variables[1],
                                              self.out.variables[0],
                                              self.out.variables[1]]))
        return L


def create_int_cluster_model():
    global int_cluster_model
    int_cluster_model = IntCluster()


def train_int_cluster_model(data, epochs):
    print("Integrated Clustering Training Started")
    with tf.device('/device:GPU:0'):
        for epoch in range(epochs):
            int_cluster_loss = []
            start = time.time()
            for (batch, (inp1, inp2, inp3)) in enumerate(data):
                batch_loss = int_cluster_model.network_learn(inp1,inp2,inp3, [0.007,0.003,0.001])
                if batch % 40 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))
                int_cluster_loss.append(batch_loss)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, np.array(int_cluster_loss).mean(),))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print("Integrated Clustering Training  Finished")



###############################################################################################





def evalate_test_data():
    eval_batch_size = 512
    en_toklist_test,es_toklist_test = load_testdata()
#     en_toklist_test = en_toklist_test[0:10]
#     es_toklist_test = es_toklist_test[0:10]
    test_bleu = get_test_bleu(en_toklist_test,es_toklist_test,eval_batch_size)
    print("Test bleu score without unlabelled data: ",test_bleu)
    test_bleu_scores = []
    with open("./test_bleu.pickle", 'rb') as handle:
        test_bleu_scores = pickle.load(handle)
    test_bleu_scores.append(test_bleu)
    with open("./test_bleu.pickle","wb") as handle:
        pickle.dump(test_bleu_scores,handle, protocol=pickle.HIGHEST_PROTOCOL)



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



def main():
    global MAX_LENGTH,VOCAB_SIZE
    train_type = sys.argv[1]
    A2L_Flag = int(sys.argv[2])
    MAX_LENGTH = int(sys.argv[3])
    VOCAB_SIZE = int(sys.argv[4])



    transformer_iterative_epoch = 3
    transformer_full_epoch = 20
    siamese_epoch = 25
    int_cluster_epoch = 1


    tf.random.set_seed(7)
    initialise_GV()



    if train_type=="full":
        print("Training LSTM Full iteration")
        load_optimizers()
        train_data = []
        with open("./train_data.pickle", 'rb') as handle:
            train_data = pickle.load(handle)


        train(train_data,transformer_full_epoch)
        encoder.save_weights('./LSTM_ckpt/encoder.h5')
        decoder.save_weights('./LSTM_ckpt/decoder.h5')

        if A2L_Flag:
            print("Training Siamese")
            create_siamese_model()
            sick_dataset = get_siamese_data()
            train_siamese(sick_dataset,siamese_epoch)
            Siamese_Model.save_weights('./Siamese_ckpt/siamese.h5')
            # train_siamese(sick_dataset,1)
            # Siamese_Model.load_weights('./Siamese_ckpt/siamese.h5')


            print("Training Integrated Clustering")
            create_int_cluster_model()
            sick_dataset2 = get_int_cluster_data()
            train_int_cluster_model(sick_dataset2, int_cluster_epoch)
            int_cluster_model.save_weights('./IntCluster_ckpt/int_cluster.h5')

        evalate_test_data()


    else:
        print("Training LSTM Incremental Iteration")
        build_model()
        load_optimizers()
        encoder.load_weights('./LSTM_ckpt/encoder.h5')
        decoder.load_weights('./LSTM_ckpt/decoder.h5')
        print("Checkpoint loaded")
        train_data = []
        with open("./train_data.pickle", 'rb') as handle:
            train_data = pickle.load(handle)
        print("Number of training samples:", len(train_data))

        train(train_data,transformer_iterative_epoch)
        encoder.save_weights('./LSTM_ckpt/encoder.h5')
        decoder.save_weights('./LSTM_ckpt/decoder.h5')

if __name__ == "__main__":
    main()
