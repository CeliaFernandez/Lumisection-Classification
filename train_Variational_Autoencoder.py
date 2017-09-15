#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:12:52 2017

@author: celiafernandezmadrazo
"""

"""
This script constructs an autoencoder model which is trained on good lumisections from data
"""
from Trial import Trial
from Variational_Autoencoder import Variational_Autoencoder

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm
import random


from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
import keras

import pickle


#%% Pre-training: Loading and preparing data

print('Loading variables...')

with open('Variables/lumisection_variables.pickle', 'rb') as f:
    goodLumi, badLumi, goodTargets, badTargets, feature_names = pickle.load(f)
    
np.random.shuffle(goodLumi)

goodLumi = goodLumi[:,range(0,len(goodLumi[0])-5)]
badLumi = badLumi[:,range(0,len(badLumi[0])-5)]
feature_names = feature_names[0:len(feature_names)-5]
    
print('-----Variables loaded-----')
      
print('------> Training only on good lumisections')

goodTrial = Trial(goodLumi, goodTargets, feature_names) # generate a Trial object
        
goodTrial.generate_trial(0.8)


#%% Generate training variables and normalize the data

x_train = goodTrial.x_train
x_test = goodTrial.x_test

# Trainign characteristics 
    
batch_size = 1
epochs = 50
original_dim = len(x_train[0])
latent_dim = 2
intermediate_dim = 24

# Adjust batch size
x_train = goodTrial.adjust_batch(x_train, batch_size)
x_test = goodTrial.adjust_batch(x_test, batch_size)

x_original = np.copy(x_test) # copy of the vector for final representation

                    
# Normalize the data
train_mean, train_sigma, train_maximum, train_minimum = goodTrial.get_normalization(x_train)
test_mean, test_sigma, test_maximum, test_minimum = goodTrial.get_normalization(x_test)

x_train = goodTrial.center_data(x_train, train_mean, train_sigma, train_maximum, train_minimum)
x_test = goodTrial.center_data(x_test, test_mean, test_sigma, test_maximum, test_minimum)

# Save variables
with open('Training/training_variables.pickle', 'wb') as f:
    pickle.dump([x_train, x_test, x_original], f)


with open('Training/training_options.pickle', 'wb') as f:
    pickle.dump([batch_size, epochs, original_dim, latent_dim, intermediate_dim], f) # pickle file
    
               
#%% Create, train and save the model

Autoencoder = Variational_Autoencoder(batch_size = batch_size, original_dim = original_dim, latent_dim = latent_dim, intermediate_dim = intermediate_dim, epochs = epochs)
Autoencoder.train(x_train, x_test)


with open('Training/loss.pickle', 'wb') as f:
    pickle.dump([Autoencoder.history.losses], f) # pickle file
    
plt.clf()
plt.figure(figsize=(6, 6))
plt.plot(list(range(0,len(Autoencoder.history.losses))), Autoencoder.history.losses)
plt.xlabel('Epochs', fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
plt.savefig('Training/loss.png', dpi = 600)



