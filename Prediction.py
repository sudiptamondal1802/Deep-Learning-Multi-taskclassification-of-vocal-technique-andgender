#!/usr/bin/env python
# coding: utf-8

# ### Imports 

# In[1]:


import os
import keras
import librosa, librosa.display
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[16]:


SAMPLE_RATE= 22050    
FILE_PATH = '/Users/sudiptamondal/Documents/QMUL/Deep learning for Audio and Music/Project/VocalSet/VocalSet11/PredictSamples'
PICKLE_DUMP = '/Users/sudiptamondal/Documents/QMUL/Deep learning for Audio and Music/Project/prediction'
PICKLE_SCALAR_DUMP = '/Users/sudiptamondal/Documents/QMUL/Deep learning for Audio and Music/Project/scaler_train_fit'


# ### Preprocess the sample data

# In[3]:


def extract_melspectrogram(audio, win_len=0.05, hop_len=0.025, n_mels=64):
    
    win_len = int(win_len*sr) #window length
    hop_len = int(hop_len*sr) #hop length
    
    #compute mel sclaed spectrogram
    spec = librosa.feature.melspectrogram(audio, sr, n_mels=n_mels, n_fft=2048, win_length=win_len, hop_length=hop_len)
    # return data format (time_len, n_mels)
    return spec.transpose((1,0))


# In[17]:


def preprocess_sample(X):
    
    #data normalisation
    #scaler = StandardScaler()
    
    # Need to use the normalisation that has been already fit in the train dataset.....check the last sem ML assignment
    #scaler.fit(X.reshape((-1,64)))   #THIS IS ONLY TEMPORARY CHANGE IT!!!!!!!!!!!!!
    my_file = open(PICKLE_SCALAR_DUMP,'rb')
    scaler = pickle.load(my_file)
    X = scaler.transform(X.reshape((-1,64)))
    X = X.reshape((-1, 500, 64)) # reverse back to the original shape
    
    #the will be used in 2D format for LSTM
    X_LSTM = X
    
    # Convert from 2D array to 3D array for each sample
    # CNN expects 3D array, becuase we have the 3rd dimension channel, for audio it will 1 channel
    X_CNN  = X[..., np.newaxis] 

    return X_LSTM, X_CNN


# In[18]:


def show_melspectrogram(X_sample):    
    #visualise the melspec for the signal
    fig, ax = plt.subplots()
    M = X_sample.transpose((1,0))  #transposing back for the ease of melspec visualisation
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time')
    ax.set(title=X_sample)
    fig.colorbar(img, ax=ax, format="%+2.f dB")


# ### Load the samples in case needed

# In[19]:


my_file = open(PICKLE_DUMP,'rb')
data = pickle.load(my_file)


# In[20]:


print(type(data))


# In[21]:


for k,v in data.items():
    print(v['filename'])


# ### Load the labels for gender and technique

# In[12]:


with open('gender_mapping_labels.json') as json_file:
    gender_mapping_labels = json.load(json_file)
with open('technique_mapping_labels.json') as json_file:
    technique_mapping_labels = json.load(json_file)


# In[13]:


technique_mapping_labels


# ### Make prediction

# In[39]:


def predict(model, sample):

    # prediction = [[0.1,0.2,....]] provides different probabilities for different classes
    prediction = model.predict(sample) #model.predict expect 4 dimensional array
    
    #either male or female
    if prediction[0] > 0.5:
        gender_label = 1
    else:
        gender_label = 0
    
    gender = gender_mapping_labels[str(gender_label)]
    #get top 3 techniques identified by the model
    top = technique_mapping_labels[str(prediction[1].argmax())]
    top_1 = technique_mapping_labels[str(prediction[1].argsort()[0][-1])]
    top_2 = technique_mapping_labels[str(prediction[1].argsort()[0][-2])]
    top_3 = technique_mapping_labels[str(prediction[1].argsort()[0][-3])]
    
    #map the index to the genre label with the index, so that instead of number we get a genre instead of 
    print("\n Prediction: Gender is {} and the top 3 techniques are {}, {}, {} and {}".format(gender,top,top_1,top_2, top_3))


# ### Execute the loop for making prediction for each sample

# In[40]:


import logging
logging.getLogger('tensorflow').disabled = True


# In[53]:


cnn_model = keras.models.load_model('model-CNN.h5')
lstm_model  = keras.models.load_model('model-LSTM.h5')


# In[55]:


for file in os.listdir(FILE_PATH):
    if not file.startswith('.'):
        print('Audio file is {}'.format(file))
        file_path = os.path.join(FILE_PATH,file)
        #load the sample data
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        melspec = extract_melspectrogram(audio)
        if(len(melspec) < 500):   #if the length of melspec is less than 500, then pad it to make to length 500
            melspec = np.pad(melspec, ((500-len(melspec),0),(0,0)))
        X = np.array(melspec)
        show_melspectrogram(X)
        X_LSTM, X_CNN = preprocess_sample(X)
        predict(cnn_model, X_CNN)


# In[ ]:




