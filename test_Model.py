#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:09:44 2017

@author: celiafernandezmadrazo
"""

"""
This script use a previous compiled model and run over the samples available 
"""

import pickle
from Trial import Trial
from Variational_Autoencoder import Variational_Autoencoder
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
import keras

# Create a new Trial object that will operate with the new input data, both not seen data and test data

print('Loading variables...')

with open('Variables/lumisection_variables.pickle', 'rb') as f:
    goodLumi, badLumi, goodTargets, badTargets, feature_names = pickle.load(f)
    
    
goodLumi = goodLumi[:,range(0,len(goodLumi[0])-5)]
badLumi = badLumi[:,range(0,len(badLumi[0])-5)]
feature_names = feature_names[0:len(feature_names)-5]    
    
with open('Training/training_variables.pickle', 'rb') as f:
    x_train, x_test, x_original = pickle.load(f)
    
print('-----Variables loaded-----')


print('------> Testing on both good and bad lumisections')

inputTrial = Trial(badLumi, badTargets, feature_names)
x_input = inputTrial.x # vector to test the autoencoder


mean_input, sigma_input, maximum_input, minimum_input = inputTrial.get_normalization(x_input)
x_input = inputTrial.center_data(x_input, mean_input, sigma_input, maximum_input, minimum_input)

#mean_newGood, sigma_newGood, maximum_newGood, minimum_newGood = inputTrial.get_normalization(newGood)
#x_newGood= inputTrial.center_data(newGood, mean_newGood, sigma_newGood, maximum_newGood, minimum_newGood)


# Load the new model with the weights and run over the Trial object samples

print('Loading encoding model...')

with open('Training/training_options.pickle', 'rb') as f:
    batch_size, epochs, original_dim, latent_dim, intermediate_dim = pickle.load(f)

x_input = inputTrial.adjust_batch(x_input, batch_size)

Autoencoder = Variational_Autoencoder(batch_size = batch_size, original_dim = original_dim, latent_dim = latent_dim, intermediate_dim = intermediate_dim, epochs = epochs)
Autoencoder.load_weights('Models/weights.h5')

# Encode both input and test samples and represent the results in the latent space

x_test_encoded = Autoencoder.encode(x_test)
x_input_encoded = Autoencoder.encode(x_input)
#x_newGood_encoded = Autoencoder.encode(x_newGood)

plt.clf()
plt.figure(figsize=(6, 6))
#plt.title('Latent space representation')
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], color = 'b', edgecolors='k', label = 'Good LS')
plt.scatter(x_input_encoded[:, 0], x_input_encoded[:, 1], color = 'r', edgecolors='k', label = 'Bad LS')
plt.legend(loc = 'upper left')
#plt.axis('off')
#plt.scatter(x_newGood_encoded[:, 0], x_newGood_encoded[:, 1], color = 'g')
#plt.show()
plt.savefig('Histograms/latent_space.png', dpi = 600)


# Decode test samples and compare the feature distribution
# The same Trial object is also valid providing that the structure of the input data and training data is the same

x_test_decoded = Autoencoder.end_to_end(x_test)

test_mean, test_sigma, test_maximum, test_minimum = inputTrial.get_normalization(x_original)
x_test_decoded = inputTrial.decenter_data(x_test_decoded, test_mean, test_sigma, test_maximum, test_minimum)


decoded_pt = [x_test_decoded[i][0] for i in range(0,len(x_test_decoded))]
test_pt = [x_original[i][0] for i in range(0,len(x_test))]

plt.clf()
plt.hist(decoded_pt, bins = np.linspace(0,70,51), histtype = 'step', color = 'r', label = 'Reconstructed input', normed = 1)
plt.hist(test_pt, bins = np.linspace(0,70,51), histtype = 'step', color = 'b', label = 'Initial input', normed = 1)
plt.legend(loc='upper right')
plt.xlabel(r'$p_T $ (GeV)', fontsize = 14)
plt.savefig('Histograms/pt_distribution.png', dpi = 600)


with open('Training/toPlot.pickle', 'wb') as f:
    pickle.dump([x_original, x_test_decoded], f) # pickle file




########## PLOT GRAPHS ########

if not os.path.exists('Histograms_check'): os.makedirs('Histograms_check')
path = 'Histograms_check/'+str(epochs)+'epochs'
if not os.path.exists(path): os.makedirs(path)
        
        #plt.clf()
        #plt.plot(list(range(0,len(self.history.losses))), self.history.losses)
        #plt.savefig(path+'/loss.png', dpi = 600)
      
features = feature_names

lim = [[0, 70], #qMUPt_mean
       [0, 500], #qMuPt_rms
       [0, 0.5], #qMuPt_q1
       [0, 5], #qMuPt_q2
       [0, 10], #qMuPt_q3
       [0, 50], #qMuPt_q4
       [0, 300000], #qMuPt_q5
       [-0.4, 0.4], #qMUEta_mean
       [1.2, 2], #qMuEta_rms
       [-16, -2], #qMuEta_q1
       [-2.25, -0.25], #qMuEta_q2
       [-2, 2], #qMuEta_q3
       [0.6, 2.2], #qMuEta_q4
       [1, 16], #qMuEta_q5
       [-0.2, 0.2], #qMUPhi_mean
       [1.7, 2], #qMuPhi_rms
       [-3.2, -3], #qMuPhi_q1
       [-2, -1], #qMuPhi_q2
       [-0.25, 0.25], #qMuPhi_q3
       [1, 2], #qMuPhi_q4
       [3, 3.2], #qMuPhi_q5
       [0, 200], #qMUEn_mean
       [0, 1000], #qMuEn_rms
       [0, 2.5], #qMuEn_q1
       [0, 10], #qMuEn_q2
       [0, 30], #qMuEn_q3
       [0, 140], #qMuEn_q4
       [0, 500000], #qMuEn_q5
       [0, 35], #qMuNVtx_mean
       [0, 12], #qMuNVtx_rms
       [0, 17], #qMuNVtx_q1
       [0, 30], #qMuNVtx_q2
       [0, 35], #qMuNVtx_q3
       [0, 40], #qMuNVtx_q4
       [0, 160], #qMuNVtx_q5
       [-0.1, 0.1], #qMuCh_mean
       [0.98, 1]] #qMuCh_rms
        
       
x_units = [r' (GeV)', #qMUPt_mean
       r' (GeV)', #qMuPt_rms
       r' (GeV)', #qMuPt_q1
       r' (GeV)', #qMuPt_q2
       r' (GeV)', #qMuPt_q3
       r' (GeV)', #qMuPt_q4
       r' (GeV)', #qMuPt_q5
       ' ', #qMUEta_mean
       ' ', #qMuEta_rms
       ' ', #qMuEta_q1
       ' ', #qMuEta_q2
       ' ', #qMuEta_q3
       ' ', #qMuEta_q4
       ' ', #qMuEta_q5
       ' ', #qMUPhi_mean
       ' ', #qMuPhi_rms
       ' ', #qMuPhi_q1
       ' ', #qMuPhi_q2
       ' ', #qMuPhi_q3
       ' ', #qMuPhi_q4
       ' ', #qMuPhi_q5
       r' (GeV)', #qMUEn_mean
       r' (GeV)', #qMuEn_rms
       r' (GeV)', #qMuEn_q1
       r' (GeV)', #qMuEn_q2
       r' (GeV)', #qMuEn_q3
       r' (GeV)', #qMuEn_q4
       r' (GeV)', #qMuEn_q5
       ' ', #qMuNVtx_mean
       ' ', #qMuNVtx_rms
       ' ', #qMuNVtx_q1
       ' ', #qMuNVtx_q2
       ' ', #qMuNVtx_q3
       ' ', #qMuNVtx_q4
       ' ', #qMuNVtx_q5
       r' ($e$)', #qMuCh_mean
       r' ($e$)'] #qMuCh_rms        

        
for feature in range(0, len(x_original[0])):
    
    x_test_filled = [x_original[i][feature] for i in range(0, len(x_original))]
    x_decoded_filled = [x_test_decoded[i][feature] for i in range(0, len(x_test_decoded))]
    
    bin_n = np.linspace(lim[feature][0], lim[feature][1], 51)
    
    
    plt.ioff()
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.hist(x_decoded_filled, bins = bin_n, histtype = 'step', color = 'r', label = 'Reconstructed input', normed = 1)
    plt.hist(x_test_filled, bins = bin_n, histtype = 'step', color = 'b', label = 'Initial input', normed = 1)
    #plt.ylabel('values')
    plt.xlabel(features[feature]+x_units[feature], fontsize = 14)
    plt.legend(loc='upper right')
    plt.savefig(path +'/'+features[feature]+'_comparison.png', dpi = 600)
    
    plt.clf()
    plt.figure(figsize=(6, 6))
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(x_decoded_filled, bins = bin_n, histtype = 'step', color = 'r', label = 'decoded')
    axarr[1].hist(x_test_filled, bins = bin_n, histtype = 'step', color = 'b', label = 'x_test')
    axarr[0].set_title('Reconstructed '+features[feature]+' distribution', fontsize = 16)
    axarr[1].set_title('Input '+features[feature]+' distribution', fontsize = 16)
    plt.xlabel(features[feature]+x_units[feature], fontsize = 14)
    plt.savefig(path +'/'+features[feature]+'_distribution.png', dpi = 600)



