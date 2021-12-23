import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K 
import warnings
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
import tensorflow as tf
import pickle as pkl

def final(x):
    weights_path = "C:/Users/Mussab/Desktop/AI/100_30_weights.h5"
    best_weights_path = "C:/Users/Mussab/Desktop/AI/100_30_best_weights.h5"

    data=pd.read_csv("C:/Users/Mussab/Desktop/AI/FinalDataAmazon.csv",nrows=100000)
    data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    temp = {'Text' : x , 'Summary' : "This is a sample summary which serves no purpose."}
    mtl = 100
    msl = 30
    data = data.append(temp, ignore_index=True)
    data.drop_duplicates(subset=['Text'],inplace=True)#dropping duplicates
    data.dropna(axis=0,inplace=True)#dropping na |Ignoring this


    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                               "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                               "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                               "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                               "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                               "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                               "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                               "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                               "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                               "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                               "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                               "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                               "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                               "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                               "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                               "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                               "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                               "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                               "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                               "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                               "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                               "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                               "you're": "you are", "you've": "you have"}


    stop_words = set(stopwords.words('english')) 
    def text_cleaner(text,num):
        newString = text.lower()
        newString = BeautifulSoup(newString, "lxml").text
        newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub('"','', newString)
        newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
        newString = re.sub(r"'s\b","",newString)
        newString = re.sub("[^a-zA-Z]", " ", newString) 
        newString = re.sub('[m]{2,}', 'mm', newString)
        if(num==0):
            tokens = [w for w in newString.split() if not w in stop_words]
        else:
            tokens=newString.split()
        long_words=[]
        for i in tokens:
            if len(i)>1:                                                 #removing short word
                long_words.append(i)   
        return (" ".join(long_words)).strip()


    #call the function
    cleaned_text = []
    for t in data['Text']:
        cleaned_text.append(text_cleaner(t,0))


    #call the function
    cleaned_summary = []
    for t in data['Summary']:
        cleaned_summary.append(text_cleaner(t,1))


    data['cleaned_text']=cleaned_text
    data['cleaned_summary']=cleaned_summary


    data.replace('', np.nan, inplace=True)
    data.dropna(axis=0,inplace=True)


    text_word_count = []
    summary_word_count = []
    # populate the lists with sentence lengths
    for i in data['cleaned_text']:
          text_word_count.append(len(i.split()))

    for i in data['cleaned_summary']:
          summary_word_count.append(len(i.split()))
    length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
    length_df.hist(bins = 30)
    plt.show()


    cnt=0
    for i in data['cleaned_summary']:
        if(len(i.split())<=15):
            cnt=cnt+1


    max_text_len = mtl
    max_summary_len = msl


    cleaned_text =np.array(data['cleaned_text'])
    cleaned_summary=np.array(data['cleaned_summary'])
    short_text=[]
    short_summary=[]
    for i in range(len(cleaned_text)):
        if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])
    df=pd.DataFrame({'text':short_text,'summary':short_summary})


    df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')


    x_train = np.array(df['text'])
    y_train = np.array(df['summary'])


    #prepare a tokenizer for reviews on training data
    x_tokenizer = Tokenizer() 
    x_tokenizer.fit_on_texts(list(x_train))


    thresh=4
    cnt=0
    tot_cnt=0
    freq=0
    tot_freq=0
    for key,value in x_tokenizer.word_counts.items():
        tot_cnt=tot_cnt+1
        tot_freq=tot_freq+value
        if(value<thresh):
            cnt=cnt+1
            freq=freq+value    


    #prepare a tokenizer for reviews on training data
    x_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
    x_tokenizer.fit_on_texts(list(x_train))
    #convert text sequences into integer sequences
    x_train_seq    =   x_tokenizer.texts_to_sequences(x_train) 
    # x_test_seq   =   x_tokenizer.texts_to_sequences(x_test)
    #padding zero upto maximum length
    x_train    =   pad_sequences(x_train_seq,  maxlen=max_text_len, padding='post')
    # x_test   =   pad_sequences(x_test_seq, maxlen=max_text_len, padding='post')
    #size of vocabulary ( +1 for padding token)
    x_vocab   =  x_tokenizer.num_words + 1


    #prepare a tokenizer for reviews on training data
    y_tokenizer = Tokenizer()   
    y_tokenizer.fit_on_texts(list(y_train))


    thresh=6
    cnt=0
    tot_cnt=0
    freq=0
    tot_freq=0
    for key,value in y_tokenizer.word_counts.items():
        tot_cnt=tot_cnt+1
        tot_freq=tot_freq+value
        if(value<thresh):
            cnt=cnt+1
            freq=freq+value


    # #prepare a tokenizer for reviews on training data
    y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
    y_tokenizer.fit_on_texts(list(y_train))
    #convert text sequences into integer sequences
    y_train_seq    =   y_tokenizer.texts_to_sequences(y_train) 
    # y_test_seq   =   y_tokenizer.texts_to_sequences(y_test) 
    #padding zero upto maximum length
    y_train    =   pad_sequences(y_train_seq, maxlen=max_summary_len, padding='post')
    # y_test   =   pad_sequences(y_test_seq, maxlen=max_summary_len, padding='post')
    #size of vocabulary
    y_vocab  =   y_tokenizer.num_words +1


    y_tokenizer.word_counts['sostok'],len(y_train)


    ind=[]
    for i in range(len(y_train)):
        cnt=0
        for j in y_train[i]:
            if j!=0:
                cnt=cnt+1
        if(cnt==2):
            ind.append(i)

    y_train=np.delete(y_train,ind, axis=0)
    x_train=np.delete(x_train,ind, axis=0)


    x_vocab = pkl.load(open("C:/Users/Mussab/Desktop/AI/x_voc_size1_100_30",'rb'))
    y_vocab = pkl.load(open("C:/Users/Mussab/Desktop/AI/y_voc_size1_100_30",'rb'))


    max_text_len=mtl
    max_summary_len=msl


    K.clear_session()
    latent_dim = 300
    embedding_dim=100
    # Encoder
    encoder_inputs = Input(shape=(max_text_len,))
    #embedding layer
    enc_emb =  Embedding(x_vocab, embedding_dim,trainable=True)(encoder_inputs)
    #encoder lstm 1
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
    #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
    #encoder lstm 3
    encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    #embedding layer
    dec_emb_layer = Embedding(y_vocab, embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])
    # Attention layer
    attn_out = Attention()([decoder_outputs, encoder_outputs])
    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    #dense layer
    decoder_dense =  TimeDistributed(Dense(y_vocab, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    try:
      model.load_weights(best_weights_path)
      print("Weights Loaded Successfully")
    except:
      print("Weights are not initialized")


    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index


    # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    # decoder_state_input_c = Input(shape=(,))
    decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))
    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs) 
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
    #attention inference
    attn_out_inf = Attention()([decoder_outputs2, decoder_hidden_state_input])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 
    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))
        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = target_word_index['sostok']
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :]) #
            sampled_token = reverse_target_word_index[sampled_token_index]
            if(sampled_token!='eostok'):
                decoded_sentence += ' '+sampled_token
            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
                stop_condition = True
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
            # Update internal states
            e_h, e_c = h, c
        return decoded_sentence


    def seq2summary(input_seq):
        newString=''
        for i in input_seq:
            if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
                newString=newString+reverse_target_word_index[i]+' '
        return newString


    def seq2text(input_seq):
        newString=''
        for i in input_seq:
            if(i!=0):
                newString=newString+reverse_source_word_index[i]+' '
        return newString

    for i in range(len(x_train)-1,len(x_train)):
      orig = seq2text(x_train[i])
      pred = decode_sequence(x_train[i].reshape(1,max_text_len))
    return pred

