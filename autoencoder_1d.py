### imports

# external modules
import sys
import os
import pandas as pd
import importlib
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

#local modules
from json_utils import *
from data_preparation import *

##TO DO arg parse

verbose = True

# note: this script assumes you have a csv file stored at the specified location,
#     containing only histograms of the specified type

histname = 'FEDTotalEventSize'
filename = 'nanodqmio_2023C_Muon0_'+histname+'_mod.csv'
datadir = '/eos/user/f/fsimone/auto_DQM/output_nanodqm/'

def mseTop10(y_true, y_pred):
    ### MSE top 10 loss function for autoencoder training
    # input arguments:
    # - y_true and y_pred: two numpy arrays of equal shape,
    #   typically a histogram and its autoencoder reconstruction.
    #   if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!
    # output:
    # - mean squared error between y_true and y_pred,
    #   where only the 10 bins with largest squared error are taken into account.
    #   if y_true and y_pred are 2D arrays, this function returns 1D array (mseTop10 for each histogram)
    top_values, _ = tf.nn.top_k(K.square(y_pred - y_true), k=10, sorted=True)
    mean=K.mean(top_values, axis=-1)
    return mean

### getting a keras model ready for training with minimal user inputs

def getautoencoder(input_size,arch,act=[],opt='adam',loss=mseTop10):
  ### get a trainable autoencoder model
  # input args:
  # - input_size: size of vector that autoencoder will operate on
  # - arch: list of number of nodes per hidden layer (excluding input and output layer)
  # - act: list of activations per layer (default: tanh)
  # - opt: optimizer to use (default: adam)
  # - loss: loss function to use (defualt: mseTop10)
  
  import math
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
  from tensorflow.keras.layers import Input, Dense
  from tensorflow.keras.models import Model, Sequential, load_model
  from keras import backend as K
  
  if len(act)==0: act = ['tanh']*len(arch)
  layers = []
  # first layer manually to set input_dim
  layers.append(Dense(arch[0],activation=act[0],input_dim=input_size))
  # rest of layers in a loop
  for nnodes,activation in zip(arch[1:],act[1:]):
    layers.append(Dense(nnodes,activation=activation))
  # last layer is decoder
  layers.append(Dense(input_size,activation='tanh'))
  autoencoder = Sequential()
  for i,l in enumerate(layers):
    #l.name = 'layer_'+str(i)
    autoencoder.add(l)
  autoencoder.compile(optimizer=opt, loss=loss)
  autoencoder.summary()
  return autoencoder

def train_simple_autoencoder(hists, nepochs=-1, modelname='', 
               batch_size=500, shuffle=False, 
               verbose=1, validation_split=0.1,
               returnhistory=False ):
  ### create and train a very simple keras model
  # the model consists of one hidden layer (with half as many units as there are input bins), 
  # tanh activation, adam optimizer and mseTop10 loss.
  # input args: 
  # - hists is a 2D numpy array of shape (nhistograms, nbins)
  # - nepochs is the number of epochs to use (has a default value if left unspecified)
  # - modelname is a file name to save the model in (default: model is not saved to a file)
  # - batch_size, shuffle, verbose, validation_split: passed to keras .fit method
  # - returnhistory: boolean whether to return the training history (e.g. for making plots)
  # returns
  # - if returnhistory is False, only the trained keras model is returned
  # - if returnhistory is True, the return type is a tuple of the form (model, history)
  input_size = hists.shape[1]
  arch = [int(hists.shape[1]/2.)]
  act = ['tanh']*len(arch)
  opt = 'adam'
  loss = mseTop10
  if nepochs<0: nepochs = int(min(40,len(hists)/400))
  model = getautoencoder(input_size,arch,act=act,opt=opt,loss=loss)
  history = model.fit(hists, hists, epochs=nepochs, batch_size=batch_size, 
            shuffle=shuffle, verbose=verbose, 
            validation_split=validation_split)
  if len(modelname)>0: model.save(modelname.split('.')[0]+'.h5')
  if not returnhistory: return model
  else: return (model, history)

if __name__ == "__main__":
  # load the file into a dataframe
  df = pd.read_csv(datadir+filename)
  if verbose:
    msg = 'INFO in autoencoder_1d:'
    msg += ' loaded a dataframe with {} rows and {} columns.'.format(len(df), len(df.columns))
    print(msg)
  
  ### filtering: select only DCS-bit on data
  ### TO-DO: filter out low statistics
  df = select_dcson(df)
  print('number of passing lumisections after DCS selection: {}'.format( len(df) ))
  
  ### preprocessing of the data: rebinning and normalizing
  X_train = get_hist_values(df)[0]  # returns np array of shape (nhists,nbins) (for 1D hists)
  X_train = rebin1D(X_train,2)
  X_train = norm1D(X_train)

  input_size = X_train.shape[1]
  arch = [int(X_train.shape[1]/2.)]
  act = ['tanh']*len(arch)
  opt = 'adam'
  loss = mseTop10
  autoencoder = getautoencoder(input_size,arch,act,opt,loss) 
