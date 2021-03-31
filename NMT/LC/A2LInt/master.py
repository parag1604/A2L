import time
import pickle
import tensorflow_datasets as tfds
import os
import random
import sys
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



VOCAB_SIZE = 2**13
MAX_LENGTH = 50
global_data_path = "../../global_data/"
GPU_NUM = '0'



def main():


    A2L_Flag = 1

    tokenizer_base_dir = global_data_path+"Tokenizer/"
    tokenizer_en_dir = tokenizer_base_dir+'tok_en_' + str(VOCAB_SIZE) + '.pickle'
    tokenizer_es_dir = tokenizer_base_dir+'tok_pt_' + str(VOCAB_SIZE) + '.pickle'


    training_data_dir = global_data_path+"training_data/"
    labelled_dir = training_data_dir+str(VOCAB_SIZE)+"/"+str(MAX_LENGTH)+"/Labelled.pickle"
    unlabelled_dir = training_data_dir+str(VOCAB_SIZE)+"/"+str(MAX_LENGTH)+"/Unlabelled.pickle"






    print("MAX_LENGTH:",MAX_LENGTH)

    code_start = datetime.datetime.now()


    print("Loading Tokenizer")
    tokenizer_src = pickle.load(open(tokenizer_en_dir, 'rb'))
    tokenizer_tar = pickle.load(open(tokenizer_es_dir, 'rb'))


    print("Loading Training data")
    start_time = time.time()


    Labelled_data = []
    Unlabelled_data = []
    with open(labelled_dir, 'rb') as handle:
        Labelled_data = pickle.load(handle)
    with open(unlabelled_dir, 'rb') as handle:
        Unlabelled_data = pickle.load(handle)

#     Labelled_data = Labelled_data[0:1000]
    Unlabelled_data = Unlabelled_data[0:500000]

    print("Total Labelled samples :",len(Labelled_data))
    print("Total Unlabelled samples :",len(Unlabelled_data))
    print("--- %s seconds ---" % (time.time() - start_time)) #166 seconds



    SAMPLE_NUM = 10000
    iteration_limit = 5
    full_data_limit = 50000
    count_samples_added = 0
    no_of_total_iterations = 0
    iteration_num=0




    with open("./test_bleu.pickle","wb") as handle:
        pickle.dump([],handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("./train_data.pickle", 'wb') as handle:
        pickle.dump(Labelled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    command = "python -u train.py full "+str(A2L_Flag)+" "+str(MAX_LENGTH)+" "+str(VOCAB_SIZE)+" "+GPU_NUM

    ret_val = os.system(command)
    if ret_val!=0:
        print("Exit Code:",ret_val)
        sys.exit("Error Occured")


    random_seed = 7
    random.seed(random_seed)

    while True:
        if no_of_total_iterations>=iteration_limit:
            break

        start = time.time()

        print("Current Working Directory:",os.getcwd())
        print("GPU_NUM:",GPU_NUM)
        print("Iteration: ", iteration_num)
        print("Newly Added Samples",count_samples_added)
        print("Full Iterations completed: ",no_of_total_iterations)
        print("Number of Labelled samples: ",len(Labelled_data))
        print("Number of Unlabelled samples: ",len(Unlabelled_data))
        test_bleu_scores = []
        with open("./test_bleu.pickle", 'rb') as handle:
            test_bleu_scores = pickle.load(handle)
        print("Test bleu scores are:",test_bleu_scores)




        iteration_num+=1


        chosen_unlabelled_data = random.sample(Unlabelled_data,SAMPLE_NUM)
        with open("./sample_data.pickle", 'wb') as handle:
            pickle.dump(chosen_unlabelled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        command = "python -u AL.py "+str(A2L_Flag)+" "+str(MAX_LENGTH)+" "+str(VOCAB_SIZE)+" "+GPU_NUM
        ret_val = os.system(command)
        if ret_val!=0:
            print("Exit Code:",ret_val)
            sys.exit("Error Occured")



        chosen_samples = []
        with open("./chosen_data.pickle", 'rb') as handle:
            chosen_samples = pickle.load(handle)


        count_samples_added += len(chosen_samples)
        print("Total Chosen Samples",len(chosen_samples))


        if count_samples_added>=full_data_limit:
            print("Training full iteration")
            Labelled_data = Labelled_data + chosen_samples
            Unlabelled_data = list(set(Unlabelled_data).difference(set(chosen_samples)))

            with open("./train_data.pickle", 'wb') as handle:
                pickle.dump(Labelled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            command = "python -u train.py full "+str(A2L_Flag)+" "+str(MAX_LENGTH)+" "+str(VOCAB_SIZE)+" "+GPU_NUM

            ret_val = os.system(command)
            if ret_val!=0:
                print("Exit Code:",ret_val)
                sys.exit("Error Occured")

            count_samples_added = 0
            no_of_total_iterations+=1
        else:
            print("Incremental training")
            train_data = random.sample(Labelled_data,len(chosen_samples)) + chosen_samples
            Labelled_data = Labelled_data + chosen_samples
            Unlabelled_data = list(set(Unlabelled_data).difference(set(chosen_samples)))

            with open("./train_data.pickle", 'wb') as handle:
                pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            command = "python -u train.py iterative "+str(A2L_Flag)+" "+str(MAX_LENGTH)+" "+str(VOCAB_SIZE)+" "+GPU_NUM

            ret_val = os.system(command)
            if ret_val!=0:
                print("Exit Code:",ret_val)
                sys.exit("Error Occured")

        print('Time taken for this iteration: {} secs\n'.format(time.time() - start))

    test_bleu_scores = []
    with open("./test_bleu.pickle", 'rb') as handle:
        test_bleu_scores = pickle.load(handle)
    print("Test bleu scores are:",test_bleu_scores)
    print("Random seed used: ",random_seed)

    code_end = datetime.datetime.now()
    print("Start time:",code_start)
    print("End time:",code_end)



if __name__ == "__main__":
    main()
