#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:45:32 2017

@author: celiafernandezmadrazo
"""

import numpy as np
import random
import functions as f
import keras
import matplotlib.pyplot as plt
import os

class Trial:
    
    def __init__(self, x, y, xi):
        
        self.total = len(x) # total number of samples
      
        self.x = x # samples atribute
        self.y = y # targets atribute
        self.features = xi # features that are included in each x
        
    def generate_trial(self, train_fraction):
        
        # train_percent is the fraction of total samples x used to train the NN
        
        random.shuffle(self.x) # Be careful if you use y -> CHANGE THE CODE
        
        index = int(self.total*train_fraction)
        self.x_train = self.x[:index]
        self.x_test = self.x[index:]
        self.y_train = self.y[:index]
        self.y_test = self.y[index:]
        
    def get_normalization(self, sampleSet):
        
        self.norm_features = len(self.features)  # exclude the charge
                                
        samples = len(sampleSet)
        
        mean = np.zeros(self.norm_features)
        sigma = np.zeros(self.norm_features)
        maximum = np.zeros(self.norm_features)
        minimum = np.zeros(self.norm_features)
        
        for feature in range(0, self.norm_features):
            
            feature_set = np.zeros(samples)
            
            for sample in range(0, samples):
                
                feature_set[sample]= sampleSet[sample][feature]
        
            mean[feature] = np.mean(feature_set)
            maximum[feature] = np.max(feature_set)
            minimum[feature] = np.min(feature_set)
            sigma[feature] = np.std(feature_set)
            
        return mean, sigma, maximum, minimum
        
            
    def center_data(self, vector, mean, sigma, maximum, minimum):
        
        normvector = np.copy(vector)
        
        for feature in range(0, self.norm_features):
            
            print(str(feature)+' has been normalized')
            
            for sample in range(0, len(vector)):
                
                normvector[sample][feature] = f.norm3(normvector[sample][feature], mean[feature], maximum[feature], minimum[feature])
             
        
        #normvector += 1 
        
        return normvector
              
    def decenter_data(self, normvector, mean, sigma, maximum, minimum):
        
        #normvector -= 1
        
        vector = np.copy(normvector)
        
        for feature in range(0, self.norm_features):
            
            
            for sample in range(0, len(vector)):
                
                vector[sample][feature] = f.denorm3(vector[sample][feature], mean[feature], maximum[feature], minimum[feature])
                
        return vector
    
    def adjust_batch(self, vector, batch_size):
        
        n = len(vector)
        d = 1
        
        for i in range(1, n):
            
            if i%batch_size == 0:
                
                d = i # high divisor
        
        return vector[:d]
                
        
        
        
    
    def show_results(self, epochs, arg):
        
        # Histogram variables 
        
        if not os.path.exists('Histograms_check'): os.makedirs('Histograms_check')
        path = 'Histograms_check/'+str(epochs)+'epochs_'+arg
        if not os.path.exists(path): os.makedirs(path)
        
        #plt.clf()
        #plt.plot(list(range(0,len(self.history.losses))), self.history.losses)
        #plt.savefig(path+'/loss.png', dpi = 600)
        
        for feature in range(0, len(self.features)):
            
            x_test_filled = [self.x_test[i][feature] for i in range(0, len(self.x_test))]
            x_decoded_filled = [self.x_decoded[i][feature] for i in range(0, len(self.x_decoded))]
            
            plt.clf()
            plt.hist(x_decoded_filled, bins = 50, histtype = 'step', color = 'r', label = 'decoded', normed = 1)
            plt.hist(x_test_filled, bins = 50, histtype = 'step', color = 'b', label = 'x_test', normed = 1)
            #plt.ylabel('values')
            plt.xlabel(self.features[feature])
            plt.legend(loc='upper right')
            plt.savefig(path +'/'+self.features[feature]+'_comparison.png', dpi = 600)
            
            plt.clf()
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].hist(x_decoded_filled, bins = 50, histtype = 'step', color = 'r', label = 'decoded')
            axarr[1].hist(x_test_filled, bins = 50, histtype = 'step', color = 'b', label = 'x_test')
            axarr[0].set_title('Decoded '+self.features[feature]+' distribution')
            axarr[1].set_title('Test '+self.features[feature]+' distribution')
            plt.savefig(path +'/'+self.features[feature]+'_distribution.png', dpi = 600)
            
            
            
            