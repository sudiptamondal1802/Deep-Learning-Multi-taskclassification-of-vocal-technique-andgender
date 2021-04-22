#!/usr/bin/env python
# coding: utf-8

# ### Preparing the dataset

# In[1]:


import  os
import shutil
import librosa, librosa.display
from string import digits
import numpy as np
import IPython.display as ipd
import pandas as pd
import json


# In[2]:


import logging
logging.getLogger('tensorflow').disabled = True


# In[3]:


import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as nnF

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# for accuracy and confusion matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[856]:


#DATASET_PATH = "./VocalSet/VocalSet11/FULL_copy"
DATASET_PATH = "./VocalSet/VocalSet11/FULL"
SAMPLE_RATE= 44100
DESTINATION = '/Users/sudiptamondal/Documents/QMUL/Deep learning for Audio and Music/Project/VocalSet/VocalSet11/PredictSamples'
PICKLE_DUMP = '/Users/sudiptamondal/Documents/QMUL/Deep learning for Audio and Music/Project/prediction'
PICKLE_SCALER_DUMP = '/Users/sudiptamondal/Documents/QMUL/Deep learning for Audio and Music/Project/scaler_train_fit'


# ### Pending tasks
# - create a few unseen samples from X_test seperate or get some samples from wild and create a new py file to iterate over the samples when makin prediction
# - confusion matrix
# 
# - Show a data sample with MFCC, if time permits

# ### Data processing

# In[54]:


def extract_melspectrogram(audio, sr=SAMPLE_RATE, win_len=0.05, hop_len=0.025, n_mels=64,n_fft=2205):
    
    win_len = int(win_len*sr) #window length
    hop_len = int(hop_len*sr) #hop length
    
    #compute mel sclaed spectrogram
    spec = librosa.feature.melspectrogram(audio, sr, n_mels=n_mels, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
    # return data format (time_len, n_mels)
    return spec.transpose((1,0))


# In[375]:


# to remove digits
remove_digits = str.maketrans('', '', digits)
    
#dictionary to store training data
data = {
    #mel spectrogram to be stored for each signal due to the different audio singal lentghs, these are the inputs as vectors
    "melspec" :[],  
    "labels_gender":[],      
    "labels_technique":[] 
    }

#melspec for each gender and technique created dynamically
melspec_1_each_technique_gender = {}

for path in os.listdir(DATASET_PATH):
    if not path.startswith('.'):
        gender_label = path.translate(remove_digits)
        subdir_path  = os.listdir(os.path.join(DATASET_PATH,path))
        for subdir in subdir_path:
            if not subdir.startswith('.'):
                vocal_techniques = os.listdir(os.path.join(DATASET_PATH,path,subdir))
                for technique in vocal_techniques:
                    if not technique.startswith('.'):
                        technique_path = os.listdir(os.path.join(DATASET_PATH,path,subdir,technique))
                        counter = 0
                        for file in technique_path:
                            if not file.startswith('.'):
                                file_path = os.path.join(DATASET_PATH,path,subdir,technique,file)
                                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                                #if librosa.get_duration(y=audio, sr=SAMPLE_RATE) > 5: #consider only files with more than 5 seconds of audio
                                melspec = extract_melspectrogram(audio)
                                # return data format (time_len, n_mels)
                                if(len(melspec) < 500):   #if the length of melspec is less than 500, then pad it to make to length 500
                                    melspec = np.pad(melspec, ((500-len(melspec),0),(0,0)))
                                    #Move some sample files from the dataset for prediction later
                                else:
                                    continue
                                
                                if counter == 0:                                        
                                    gender_technique = str(gender_label+'_'+technique)
                                    melspec_1_each_technique_gender[gender_technique] = {}
                                    melspec_1_each_technique_gender[gender_technique]["filename"] = file
                                    melspec_1_each_technique_gender[gender_technique]["audio_signal"] = list(audio)
                                    melspec_1_each_technique_gender[gender_technique]["melspec"] = list(melspec)
                                    #shutil.move(file_path,DESTINATION)
                                #else:
                                # store melspec 
                                data["melspec"].append(melspec[:500])
                                # store labels
                                data["labels_gender"].append(gender_label) 
                                data["labels_technique"].append(technique)
                                        
                                counter += 1                                


# ### Copy the dictionary into pickle

# In[376]:


import pickle
my_file = open(PICKLE_DUMP, 'wb')
my_file = pickle.dump(melspec_1_each_technique_gender, my_file)


# ### Analyse the samples for each gender and technique combination

# In[377]:


###load file
my_file = open(PICKLE_DUMP,'rb')
melspec_1_each_technique_gender = pickle.load(my_file)


# In[378]:


def show_waveplot(key):
    sample_audio       = np.array(melspec_1_each_technique_gender[key]["audio_signal"])   
    sample_filename    = np.array(melspec_1_each_technique_gender[key]["filename"]) 
    
    print('The filenam is {} and the labels are {}'.format(sample_filename,key))
    
    #play the sample audio
    sample_audio_element_url = ipd.Audio(sample_audio, rate=SAMPLE_RATE)
    ipd.display(sample_audio_element_url)
    
    #visualise the signal
    librosa.display.waveplot(sample_audio, sr=SAMPLE_RATE)
    #specify the label for the x and y axis
    plt.title(key)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig(key+'_waveform')
    plt.show()


# In[379]:


def show_melspectrogram(key):
    sample_melspec  = np.array(melspec_1_each_technique_gender[key]["melspec"])    
    
    #visualise the melspec for the signal
    fig, ax = plt.subplots()
    M = sample_melspec.transpose((1,0))  #transposing back for the ease of melspec visualisation
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time')
    ax.set(title=key)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(key)


# In[380]:


for key in melspec_1_each_technique_gender.keys():
    show_waveplot(key)


# In[381]:


for key in melspec_1_each_technique_gender.keys():
    show_melspectrogram(key)


# ### Data preparation for the Deep learning models

# In[382]:


label_gender    = np.array(data["labels_gender"])
label_techinque = np.array(data["labels_technique"])


# In[387]:


len(label_techinque)


# In[383]:


from collections import Counter
gender_techinque = zip(data["labels_gender"],data["labels_technique"])
gender_technique_data = Counter(gender_techinque)


# In[384]:


gender_technique_data


# In[385]:


gender_technique_list = []
for key in gender_technique_data.keys():
    gender_technique = key[0]+'_' + key[1]
    gender_technique_list.append(gender_technique)


# In[386]:


#Frequency of labels in the dataset
f, ax = plt.subplots(figsize=(15,5))
plt.bar(gender_technique_list, gender_technique_data.values(),align='center',width=0.8)
plt.xticks(rotation=90,fontsize=14)
plt.yticks(fontsize=14)
ax.legend()


# In[388]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def labelencoder(x):
    label_encoder = LabelEncoder()  
    x_label_encoded = label_encoder.fit_transform(x)
    
    return x_label_encoded
    
def onehotencoder(x_encoded):
    onehot_encoder = OneHotEncoder(sparse=False)
    
    x_encoded = x_encoded.reshape(len(x_encoded), 1)
    x_onehot_encoded = onehot_encoder.fit_transform(x_encoded)
    
    return x_onehot_encoded

def labelencoder_mapping(x):
    label_encoder = LabelEncoder()  
    x_label_encoded = label_encoder.fit_transform(x)
    x_mapping = {l: i for i, l in enumerate(label_encoder.classes_)}
    return x_mapping


# In[389]:


gender_label = labelencoder(label_gender)

technique_label = labelencoder(label_techinque)

y = np.column_stack((gender_label,technique_label))
print(gender_label.shape)
print(technique_label.shape)
print(y.shape)


# In[390]:


gender_mapping = labelencoder_mapping(label_gender)
print(gender_mapping)

technique_mapping = labelencoder_mapping(label_techinque)
print(technique_mapping)

num_of_techniques = len(technique_mapping.keys())
print('Number of techniques: {}'.format(num_of_techniques))


# In[391]:


X = np.array(data["melspec"])
print(X.shape)


# In[392]:


print(X[0])
print(gender_label[0])
print(technique_label[0])
print(y[0])


# In[857]:


def data_normalisation(train_data, valid_data,test_data):
    # data normalisation
    scaler = StandardScaler()
    # compute normalisation parameters based on the training data 
    scaler.fit(train_data.reshape((-1,64)))
    scaler_train_fit = scaler.fit(train_data.reshape((-1,64)))
    my_file = open(PICKLE_SCALER_DUMP, 'wb')
    my_file = pickle.dump(scaler_train_fit, my_file)
    print(scaler.mean_)

    # normalise the training data with the computed parameters
    train_data = scaler.transform(train_data.reshape((-1,64)))
    train_data = train_data.reshape((-1, 500, 64)) # reverse back to the original shape
    #print(train_data[0])

    # normalise the validation data with the computed parameters
    valid_data = scaler.transform(valid_data.reshape((-1,64)))
    valid_data = valid_data.reshape((-1, 500, 64)) # reverse back to the original shape
    #print(valid_data[0])

    # normalise the test data with the computed parameters
    test_data = scaler.transform(test_data.reshape((-1,64)))
    test_data = test_data.reshape((-1, 500, 64)) # reverse back to the original shape
    #print(test_data[0])
    
    return train_data, valid_data,test_data


# In[858]:


def prepare_datasets(test_size, validation_size):

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)
    
    # create train/validation split to split the training data into train and validation set
    X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train, test_size=validation_size,shuffle=True)
    
    X_train, X_validation,X_test = data_normalisation(X_train, X_validation,X_test)
    
    #prepare dataset for LSTM model as it requires 2D data
    X_train_LSTM      = X_train
    X_validation_LSTM = X_validation
    X_test_LSTM       = X_test
    
    # Convert from 2D array to 3D array for each sample
    # CNN expects 3D array, becuase we have the 3rd dimension channel, for audio it will 1 channel
    X_train_CNN        = X_train[..., np.newaxis] 
    X_validation_CNN   = X_validation[..., np.newaxis]
    X_test_CNN         = X_test[..., np.newaxis]
    
    return X_train_LSTM, X_validation_LSTM, X_test_LSTM, X_train_CNN, X_validation_CNN, X_test_CNN, y_train, y_validation, y_test


# In[859]:


# create train, validation and test sets for both LSTM and CNN models
X_train_LSTM, X_validation_LSTM, X_test_LSTM, X_train_CNN, X_validation_CNN, X_test_CNN, y_train, y_validation, y_test = prepare_datasets(0.1, 0.2)


# In[398]:


print(X_train_CNN.shape)
print(X_validation_CNN.shape)
print(X_test_CNN.shape)


# In[399]:


print(X_train_LSTM.shape)
print(X_validation_LSTM.shape)
print(X_test_LSTM.shape)


# In[400]:


print(y_train.shape)
print(y_validation.shape)
print(y_test.shape)


# In[401]:


y_train_gender = y_train[:,0]
y_train_technique = y_train[:,1]

y_validation_gender = y_validation[:,0]
y_validation_technique = y_validation[:,1]

y_test_gender = y_test[:,0]
y_test_technique = y_test[:,1]


# In[402]:


y_test[0:10]


# ### Imports for Model

# In[677]:


import keras
import numpy as np
from keras.layers import Input, Dense, LSTM, Conv2D,MaxPooling2D,Flatten,Dropout,Activation,BatchNormalization,Bidirectional
from keras import backend as K
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# ### Model 1 - CNN

# ### Build the model

# In[617]:


def CNN_model(input_shape):
    
        filter_size  = (3,3)
        maxpool_size = (3,3)
        #stride       = (2,2)
        dropout      = 0.4
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        #Convolutional layer 1
        conv1       = Conv2D(64, kernel_size=filter_size, padding="same",activation='relu')(inputs)
        activation1 = Activation('relu')(conv1)
        maxpool1    = MaxPooling2D(pool_size=maxpool_size)(activation1)
        dropout1    = Dropout(dropout)(maxpool1)
        
        #Convolutional layer 2
        conv2       = Conv2D(32, kernel_size=filter_size, padding="same",activation='relu')(dropout1)
        activation2 = Activation('relu')(conv2)
        maxpool2    = MaxPooling2D(pool_size=maxpool_size)(activation2)
        dropout2    = Dropout(dropout)(maxpool2)
        
        #Convolutional layer 3
        conv3       = Conv2D(16, kernel_size=filter_size, padding="same",activation='relu')(dropout2)
        activation3 = Activation('relu')(conv3)
        maxpool3    = MaxPooling2D(pool_size=maxpool_size)(activation3)
        dropout3    = Dropout(dropout)(maxpool3)
        
        # flatten the output and feed it into dense layer
        flatten = Flatten()(dropout3)
        dense1   = Dense(256)(flatten)
        activation4 = Activation('relu')(dense1)
        dens2  = Dense(128)(activation4)
        activation5 = Activation('relu')(dens2)
        final_dropout = Dropout(dropout)(activation5)
        
        gender_branch     = Dense(1, activation='sigmoid', name='gender_output')(final_dropout)
        technique_branch  = Dense(num_of_techniques, activation='softmax', name='technique_output')(final_dropout)
        
        model = Model(inputs = inputs,outputs = [gender_branch, technique_branch])
        
        return model


# In[618]:


#input shape = batch, time_len, melspec
input_shape = (X_train_CNN.shape[1], X_train_CNN.shape[2], X_train_CNN.shape[3])
model1 = CNN_model(input_shape)


# In[619]:


model1.summary()


# ### Plot the model

# In[620]:


from IPython.display import SVG
from keras.utils import vis_utils
SVG(vis_utils.model_to_dot(model1, show_shapes=True, show_layer_names=True, dpi=60).create(prog='dot', format='svg'))


# In[654]:


pip install visualkeras


# In[676]:


import visualkeras
from tensorflow.python.keras.layers import InputLayer,Conv2D,Activation, Dropout, MaxPooling2D,Flatten, Dense
from collections import defaultdict

model = keras.models.load_model('model-CNN.h5')

color_map = defaultdict(dict)
color_map[InputLayer]['fill'] = 'gray'
color_map[Conv2D]['fill'] = 'orange'
color_map[Activation]['fill'] = 'pink'
color_map[Dropout]['fill'] = 'red'
color_map[MaxPooling2D]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'
color_map[Dense]['fill'] = 'blue'
visualkeras.layered_view(model, color_map=color_map,to_file='CNN.png',legend=True).show()


# In[624]:


# compile model
opt = RMSprop(learning_rate=0.0001)
model1.compile(optimizer=opt, 
               loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
               metrics=['accuracy'])


# In[625]:


#for saving the best model
callbacks1 = [
    ReduceLROnPlateau(), 
    #EarlyStopping(patience=4)
    ModelCheckpoint(filepath='model-CNN2.h5', save_best_only=True)
]


# ### Train model

# In[626]:


history = model1.fit(X_train_CNN,
                     [y_train_gender,y_train_technique],
                      validation_data=(X_validation_CNN, [y_validation_gender,y_validation_technique]),
                      batch_size=64,
                      epochs=200,
                      callbacks=callbacks1
                     )


# ### Test the model

# In[628]:


#Evaluate on test data
cnn_model = keras.models.load_model('model-CNN2.h5')
results = cnn_model.evaluate(X_test_CNN, [y_test_gender,y_test_technique])


# In[630]:


print('\nGender loss is {} and Gender accuracy is {}'.format(results[1],results[3]))
print('\nTechnique loss is {} and Technique accuracy is {}'.format(results[2],results[4]))


# In[413]:


# your code goes here
from plot_keras_history import plot_history
import matplotlib.pyplot as plt

plot_history(history.history, path="CNN.png")
plt.show()


# In[629]:


# your code goes here
from plot_keras_history import plot_history
import matplotlib.pyplot as plt

#plot loss vs epoch for Gender
plt.plot(history.history['gender_output_loss']) 
plt.plot(history.history['val_gender_output_loss']) 
plt.title('CNN Model loss for Gender')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right')
plt.savefig('loss_gender_cnn.png')
plt.show()


#plot loss vs epoch for Technique
plt.plot(history.history['technique_output_loss']) 
plt.plot(history.history['val_technique_output_loss']) 
plt.title('CNN Model loss for Technique')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right') 
plt.savefig('loss_technique_cnn.png')
plt.show()



#plot accuracy vs epoch for Gender
plt.plot(history.history['gender_output_accuracy']) 
plt.plot(history.history['val_gender_output_accuracy']) 
plt.title('CNN Model accuracy for Gender')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy'], loc='lower right') 
plt.savefig('accuracy_gender_cnn.png')
plt.show()


#plot accuracy vs epoch for Technique
plt.plot(history.history['technique_output_accuracy']) 
plt.plot(history.history['val_technique_output_accuracy']) 
plt.title('CNN Model accuracy for Technique')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy'], loc='lower right') 
plt.savefig('accuracy_technique_cnn.png')
plt.show()


# ### Make predictions for sample

# In[443]:


gender_mapping_labels = dict((int(v),k) for k,v in gender_mapping.items())
technique_mapping_labels = dict((int(v),k) for k,v in technique_mapping.items())


# In[444]:


print(gender_mapping_labels)
print(technique_mapping_labels)


# In[445]:


with open('gender_mapping_labels.json', 'w') as fp:
    json.dump(gender_mapping_labels, fp)
with open('technique_mapping_labels.json', 'w') as fp:
    json.dump(technique_mapping_labels, fp)


# In[446]:


with open('gender_mapping_labels.json') as json_file:
    gender_mapping_labels = json.load(json_file)
with open('technique_mapping_labels.json') as json_file:
    technique_mapping_labels = json.load(json_file)
print(gender_mapping_labels)
print(technique_mapping_labels)


# In[447]:


def predict(model, X,y):
    X = X[np.newaxis, ...] #inserting new axis at the beginning of the array and then copy paste the rest
    
    # prediction = [[0.1,0.2,....]] provides different probabilities for different classes
    prediction = cnn_model.predict(X) #model.predict expect 4 dimensional array
    
    #either male or female
    if prediction[0] > 0.5:
        gender_label = 1 #male
    else:
        gender_label = 0 #female
    
    gender = gender_mapping_labels[str(gender_label)]
    #get top 3 techniques identified by the model
    #top = technique_mapping_labels[prediction[1].argmax()]
    top_1 = technique_mapping_labels[str(prediction[1].argsort()[0][-1])]
    top_2 = technique_mapping_labels[str(prediction[1].argsort()[0][-2])]
    top_3 = technique_mapping_labels[str(prediction[1].argsort()[0][-3])]
    
    
    y_gender = gender_mapping_labels[str(y[0])]
    y_technique = technique_mapping_labels[str(y[1])]
    
    #map the index to the genre label with the index, so that instead of number we get a genre instead of 
    print("\n Expected: Gender is {} and Technique is {}".format(y_gender,y_technique))
    print("\n Prediction: Gender is {} and the top 3 techniques are {}, {} and {}".format(gender,top_1,top_2, top_3))


# In[633]:


# make prediction on a sample - Inference
X_sample = X_test_CNN[10]
y_sample = y_test[10]

cnn_model = keras.models.load_model('model-CNN2.h5')
predict(cnn_model, X_sample, y_sample)


# In[634]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[635]:


y_pred = cnn_model.predict(X_test_CNN)


# In[636]:


y_test_df = pd.DataFrame(y_test,columns = ['Gender','Technique'])


# In[637]:


#y_test_df.iloc[:,1]


# In[638]:


predicted_gender = [1 if  i > 0.5 else 0 for i in y_pred[0]]
predicted_technique = y_pred[1].argmax(axis=1)


# In[639]:


#set(predicted_technique)


# In[640]:


#y_test_df.iloc[:,1].unique()


# In[641]:


#matrix_gender = sklearn.metrics.confusion_matrix(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender))


# In[642]:


#matrix_technique = sklearn.metrics.confusion_matrix(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique))


# In[643]:


#matrix_technique


# In[644]:


print(gender_mapping.keys())
print(technique_mapping)


# In[645]:


# my_list = [0,1]
# _labels = []
# for k,v in gender_mapping.items():
#     if v in my_list:
#         _labels.append(k)


# In[646]:


# fig, ax = plt.subplots(figsize=(5,5)) 
# sns.heatmap(matrix_gender, linewidths=1, annot=True, ax=ax, fmt='g',
#             xticklabels=gender_mapping.keys(),yticklabels=gender_mapping.keys())
# plt.xlabel("Predicted labels",fontsize=12)
# plt.ylabel("True labels",fontsize=12)


# In[647]:


# fig, ax = plt.subplots(figsize=(16,16)) 
# sns.heatmap(matrix_technique, linewidths=1, annot=True, ax=ax, fmt='g')
#            #xticklabels=technique_mapping.keys(),yticklabels=technique_mapping.keys())
# plt.xlabel("Predicted labels",fontsize=12)
# plt.ylabel("True labels",fontsize=12)


# In[648]:


def confusion_matrix_heatmap(y_test, preds,saveas,gender=True):
    """Function to plot a confusion matrix"""
    labels = list(set(y_test))   # get the labels in the y_test
    cm = confusion_matrix(y_test, preds, labels)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    #fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    _labels = []
    
    if gender==True:
        for k,v in gender_mapping.items():
            if v in labels:
                _labels.append(k)
    else:
        for k,v in technique_mapping.items():
            if v in labels:
                _labels.append(k)
        
    ax.set_xticklabels(_labels, rotation=45)
    ax.set_yticklabels(_labels)

    thresh = cm.max() / 2.
    for i in range(len(cm)):
        for j in range(len(cm)):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted',fontsize=12)
    plt.ylabel('True',fontsize=12)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.savefig(saveas)


# In[649]:


confusion_matrix_heatmap(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender),'gender_CNN.png',gender=True)


# In[650]:


confusion_matrix_heatmap(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique),'technique_CNN2.png',gender=False)


# In[531]:


from sklearn import metrics


# In[651]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender)))
print('Accuracy for gender: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender))*100)),'%')


# In[652]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique)))
print('Accuracy for vocal technique: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique))*100)),'%')


# ### Model 2 - LSTM

# In[ ]:





# In[545]:


def LSTM_model(input_shape):
    
        filter_size  = (3,3)
        maxpool_size = (3,3)
        #stride       = (2,2)
        dropout      = 0.3
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        #LSTM layer 1
        lstm1  = LSTM(128,input_shape=input_shape, return_sequences=True)(inputs)
        lstm2  = LSTM(64, return_sequences=True)(lstm1)
        lstm3  = LSTM(64)(lstm2)
        
        # flatten the output and feed it into dense layer
        dense   = Dense(64)(lstm3)
        activation = Activation('relu')(dense)
        final_dropout = Dropout(dropout)(activation)
        
        gender_branch     = Dense(1, activation='sigmoid', name='gender_output')(final_dropout)
        technique_branch  = Dense(num_of_techniques, activation='softmax', name='technique_output')(final_dropout)
        
        model = Model(inputs = inputs,outputs = [gender_branch, technique_branch])
        
        return model


# In[546]:


X_train_LSTM.shape


# In[547]:


#input shape = batch, time_len, melspec
input_shape = (X_train_LSTM.shape[1], X_train_LSTM.shape[2])
model2 = LSTM_model(input_shape)


# In[548]:


model2.summary()


# In[605]:


SVG(vis_utils.model_to_dot(model2, show_shapes=True, show_layer_names=True, dpi=60).create(prog='dot', format='svg'))


# In[549]:


# compile model
opt = RMSprop(learning_rate=0.0001)
model2.compile(optimizer=opt, 
               loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
               metrics=['accuracy'])


# In[550]:


#for saving the best model
callbacks2 = [
    ReduceLROnPlateau(), 
    #EarlyStopping(patience=4)
    ModelCheckpoint(filepath='model-LSTM.h5', save_best_only=True)
]


# In[602]:


history2 = model2.fit(X_train_LSTM,
                     [y_train_gender,y_train_technique],
                      validation_data=(X_validation_LSTM, [y_validation_gender,y_validation_technique]),
                      batch_size=32,
                      epochs=200,
                      callbacks=callbacks2
                     )


# In[557]:


#Evaluate on test data
lstm_model = keras.models.load_model('model-LSTM.h5')
results = lstm_model.evaluate(X_test_LSTM, [y_test_gender,y_test_technique])


# In[558]:


print('\nGender loss is {} and Gender accuracy is {}'.format(results[1],results[3]))
print('\nTechnique loss is {} and Technique accuracy is {}'.format(results[2],results[4]))


# In[600]:


# your code goes here
from plot_keras_history import plot_history
import matplotlib.pyplot as plt

#print(history2.history.keys())

#plot loss vs epoch for Gender
plt.plot(history2.history['gender_output_loss']) 
plt.plot(history2.history['val_gender_output_loss']) 
plt.title('LSTM Model loss for Gender')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right') 
plt.savefig('loss_gender_lstm.png')
plt.show()


#plot loss vs epoch for Technique
plt.plot(history2.history['technique_output_loss']) 
plt.plot(history2.history['val_technique_output_loss']) 
plt.title('LSTM Model loss for Technique')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right') 
plt.savefig('loss_technique_lstm.png')
plt.show()



#plot accuracy vs epoch for Gender
plt.plot(history2.history['gender_output_accuracy']) 
plt.plot(history2.history['val_gender_output_accuracy']) 
plt.title('LSTM Model accuracy for Gender')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy'], loc='lower right') 
plt.savefig('accuracy_gender_lstm.png')
plt.show()


#plot accuracy vs epoch for Technique
plt.plot(history2.history['technique_output_accuracy']) 
plt.plot(history2.history['val_technique_output_accuracy']) 
plt.title('LSTM Model accuracy for Technique')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy'], loc='lower right') 
plt.savefig('accuracy_technique_lstm.png')
plt.show()


# In[223]:


with open('gender_mapping_labels.json') as json_file:
    gender_mapping_labels = json.load(json_file)
with open('technique_mapping_labels.json') as json_file:
    technique_mapping_labels = json.load(json_file)
print(gender_mapping_labels)
print(technique_mapping_labels)


# In[232]:


def predict_LSTM(model, X,y):
    
    X = X[np.newaxis, ...]
    # prediction = [[0.1,0.2,....]] provides different probabilities for different classes
    prediction = model.predict(X) 
    
    #either male or female
    if prediction[0] > 0.5:
        gender_label = 1
    else:
        gender_label = 0
    
    gender = gender_mapping_labels[str(gender_label)]
    #get top 3 techniques identified by the model
    #top = technique_mapping_labels[prediction[1].argmax()]
    top_1 = technique_mapping_labels[str(prediction[1].argsort()[0][-1])]
    top_2 = technique_mapping_labels[str(prediction[1].argsort()[0][-2])]
    top_3 = technique_mapping_labels[str(prediction[1].argsort()[0][-3])]
    
    
    y_gender = gender_mapping_labels[str(y[0])]
    y_technique = technique_mapping_labels[str(y[1])]
    
    #map the index to the genre label with the index, so that instead of number we get a genre instead of 
    print("\n Expected: Gender is {} and Technique is {}".format(y_gender,y_technique))
    print("\n Prediction: Gender is {} and the top 3 techniques are {}, {} and {}".format(gender,top_1,top_2, top_3))


# In[233]:


# make prediction on a sample - Inference
X_sample = X_test[10]
y_sample = y_test[10]

lstm_model = keras.models.load_model('model-LSTM.h5')
predict_LSTM(lstm_model, X_sample, y_sample)


# ### Model prediction for test data

# In[595]:


lstm_model = keras.models.load_model('model-LSTM.h5')


# In[596]:


y_pred = lstm_model.predict(X_test_LSTM)
y_test_df = pd.DataFrame(y_test,columns = ['Gender','Technique'])
predicted_gender = [1 if  i > 0.5 else 0 for i in y_pred[0]]
predicted_technique = y_pred[1].argmax(axis=1)


# In[597]:


confusion_matrix_heatmap(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender),'gender_lstm.png',gender=True)


# In[598]:


confusion_matrix_heatmap(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique),'technique_lstm.png',gender=False)


# In[570]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender)))
print('Accuracy for gender: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender))*100)),'%')


# In[571]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique)))
print('Accuracy for vocal technique: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique))*100)),'%')


# ### BiLSTM with self-attention

# In[682]:


pip install keras-self-attention


# In[685]:


from keras_self_attention import SeqSelfAttention,SeqWeightedAttention


# In[692]:


def BiLSTM_model(input_shape):
    
        dropout      = 0.4
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        bilstm1  = Bidirectional(LSTM(128,input_shape=input_shape, return_sequences=True))(inputs)
        flatten  = Flatten()(bilstm1)
        
        # flatten the output and feed it into dense layer
        dense   = Dense(64)(flatten)
        activation = Activation('relu')(dense)
        final_dropout = Dropout(dropout)(activation)
        
        gender_branch     = Dense(1, activation='sigmoid', name='gender_output')(final_dropout)
        technique_branch  = Dense(num_of_techniques, activation='softmax', name='technique_output')(final_dropout)
        
        model = Model(inputs = inputs,outputs = [gender_branch, technique_branch])
        
        return model


# In[693]:


input_shape = (X_train_LSTM.shape[1], X_train_LSTM.shape[2])
model3 = BiLSTM_model(input_shape)


# In[701]:


model3.summary()


# In[694]:


opt = RMSprop(learning_rate=0.0001)
model3.compile(optimizer=opt, 
               loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
               metrics=['accuracy'])


# In[695]:


callbacks3 = [
    ReduceLROnPlateau(), 
    #EarlyStopping(patience=4)
    ModelCheckpoint(filepath='model-biLSTM.h5', save_best_only=True)
]


# In[696]:


history3 = model3.fit(X_train_LSTM,
                     [y_train_gender,y_train_technique],
                      validation_data=(X_validation_LSTM, [y_validation_gender,y_validation_technique]),
                      batch_size=32,
                      epochs=200,
                      callbacks=callbacks3
                     )


# In[697]:


bilstm_model = keras.models.load_model('model-biLSTM.h5')
results = bilstm_model.evaluate(X_test_LSTM, [y_test_gender,y_test_technique])


# In[698]:


y_pred = bilstm_model.predict(X_test_LSTM)
y_test_df = pd.DataFrame(y_test,columns = ['Gender','Technique'])
predicted_gender = [1 if  i > 0.5 else 0 for i in y_pred[0]]
predicted_technique = y_pred[1].argmax(axis=1)


# In[699]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender)))
print('Accuracy for gender: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender))*100)),'%')


# In[700]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique)))
print('Accuracy for vocal technique: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique))*100)),'%')


# ### BiLSTM with attention

# In[ ]:


def BiLSTMattention_model(input_shape):
    
        dropout      = 0.4
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        bilstm1  = Bidirectional(LSTM(128,input_shape=input_shape, return_sequences=True))(inputs)
        weightedattention = SeqWeightedAttention()(bilstm1)
        flatten  = Flatten()(weightedattention)
        
        # flatten the output and feed it into dense layer
        dense   = Dense(64)(flatten)
        activation = Activation('relu')(dense)
        final_dropout = Dropout(dropout)(activation)
        
        gender_branch     = Dense(1, activation='sigmoid', name='gender_output')(final_dropout)
        technique_branch  = Dense(num_of_techniques, activation='softmax', name='technique_output')(final_dropout)
        
        model = Model(inputs = inputs,outputs = [gender_branch, technique_branch])
        
        return model


# In[ ]:


input_shape = (X_train_LSTM.shape[1], X_train_LSTM.shape[2])
model4 = BiLSTM_model(input_shape)


# In[ ]:


model4.summary()


# In[ ]:


opt = RMSprop(learning_rate=0.0001)
model4.compile(optimizer=opt, 
               loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
               metrics=['accuracy'])


# In[ ]:


callbacks4 = [
    ReduceLROnPlateau(), 
    #EarlyStopping(patience=4)
    ModelCheckpoint(filepath='model-biLSTMattn.h5', save_best_only=True)
]


# In[ ]:


history4 = model4.fit(X_train_LSTM,
                     [y_train_gender,y_train_technique],
                      validation_data=(X_validation_LSTM, [y_validation_gender,y_validation_technique]),
                      batch_size=32,
                      epochs=200,
                      callbacks=callbacks4
                     )


# In[ ]:


bilstmattn_model = keras.models.load_model('model-biLSTM.h5')
results = bilstmattn_model.evaluate(X_test_LSTM, [y_test_gender,y_test_technique])


# In[ ]:


y_pred = bilstmattn_model.predict(X_test_LSTM)
y_test_df = pd.DataFrame(y_test,columns = ['Gender','Technique'])
predicted_gender = [1 if  i > 0.5 else 0 for i in y_pred[0]]
predicted_technique = y_pred[1].argmax(axis=1)


# In[ ]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender)))
print('Accuracy for gender: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender))*100)),'%')


# In[ ]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique)))
print('Accuracy for vocal technique: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique))*100)),'%')


# ### CNN with Resnet

# In[762]:


from keras.applications.resnet50 import ResNet50


# In[766]:


resnet = ResNet50()


# In[767]:


resnet.summary()


# In[838]:


def CNNResnet50_model(input_shape):
        
        input_tensor = Input(shape=(500, 64, 3))
        inputs = Input(shape=input_shape)
        conv1 = Conv2D(3,(3,3),padding='same')(inputs)  
        
        resnet = ResNet50(weights='imagenet',include_top=True,input_tensor=input_tensor)   
        outputs = resnet(conv1)

        
        gender_branch     = Dense(1, activation='sigmoid', name='gender_output')(outputs)
        technique_branch  = Dense(num_of_techniques, activation='softmax', name='technique_output')(outputs)
        
        model = Model(inputs = inputs,outputs = [gender_branch, technique_branch])
        
        return model


# In[839]:


input_shape = (X_train_CNN.shape[1], X_train_CNN.shape[2], X_train_CNN.shape[3])
input_shape


# In[840]:


model_resnet = CNNResnet50_model(input_shape)


# In[841]:


model_resnet.summary()


# In[842]:


opt = RMSprop(learning_rate=0.0001)
model_resnet.compile(optimizer=opt, 
               loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
               metrics=['accuracy'])


# In[843]:


callbacks_model_resnet = [
    ReduceLROnPlateau(), 
    #EarlyStopping(patience=4)
    ModelCheckpoint(filepath='model_resnet.h5', save_best_only=True)
]


# In[847]:


history_model_resnet = model_resnet.fit(X_train_CNN,
                     [y_train_gender,y_train_technique],
                      validation_data=(X_validation_CNN, [y_validation_gender,y_validation_technique]),
                      batch_size=64,
                      epochs=200,
                      callbacks=callbacks_model_resnet
                     )


# In[850]:


model_resnet = keras.models.load_model('model_resnet.h5')


# In[852]:


results = model_resnet.evaluate(X_test_LSTM, [y_test_gender,y_test_technique])


# In[853]:


y_pred = model_resnet.predict(X_test_LSTM)
y_test_df = pd.DataFrame(y_test,columns = ['Gender','Technique'])
predicted_gender = [1 if  i > 0.5 else 0 for i in y_pred[0]]
predicted_technique = y_pred[1].argmax(axis=1)


# In[854]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender)))
print('Accuracy for gender: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,0]), np.array(predicted_gender))*100)),'%')


# In[855]:


print(metrics.classification_report(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique)))
print('Accuracy for vocal technique: ',str.format('{0:.2f}',(metrics.accuracy_score(np.array(y_test_df.iloc[:,1]), np.array(predicted_technique))*100)),'%')


# In[ ]:




