#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:56:45 2017

@author: celiafernandezmadrazo
"""

from Trial import Trial

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm


from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
import keras

# Select Theano as backend for Keras using enviroment variable `KERAS_BACKEND`
from os import environ
environ['KERAS_BACKEND'] = 'theano'


class Variational_Autoencoder:
    
    def __init__(self, batch_size, original_dim, latent_dim, intermediate_dim, epochs):
        
        self.batch_size = batch_size
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.epsilon_std = 1.0
        
        
    def train(self, x_train, x_test):
        
        x = Input(batch_shape=(self.batch_size, self.original_dim))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        h = Dense(self.intermediate_dim//2, activation='relu')(h)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        
        
        class LossHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []
        
            def on_epoch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
        
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        
        decoder_h = Dense(self.intermediate_dim//2, activation='relu')
        decoder_h2 = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        h2_decoded = decoder_h2 (h_decoded)
        x_decoded_mean = decoder_mean(h2_decoded)
        
        vae = Model(x, x_decoded_mean)
        
#        def vae_loss(x, x_decoded_mean):  PREVIOUS
#            xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#            return K.mean(xent_loss + kl_loss)
        
        def vae_loss(x, x_decoded_mean):
            xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            #xent_loss = K.sum(K.binary_crossentropy(x_decoded_mean, x), axis=1)
            #kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)
            
            return K.mean(xent_loss + kl_loss)
            #return xent_loss + kl_loss
        
        vae.compile(optimizer='sgd', loss='binary_crossentropy')
        
        self.history = LossHistory()
        
        vae.fit(x_train, x_train,
        shuffle=True,
        epochs=self.epochs,
        batch_size=self.batch_size,
        validation_data=(x_test, x_test),
        callbacks=[self.history])
        
        vae.save_weights('Models/weights.h5')
        
    def load_weights(self, weights):
        
        x = Input(batch_shape=(self.batch_size, self.original_dim))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        h = Dense(self.intermediate_dim//2, activation='relu')(h)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        decoder_h = Dense(self.intermediate_dim//2, activation='relu')
        decoder_h2 = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        h2_decoded = decoder_h2 (h_decoded)
        x_decoded_mean = decoder_mean(h2_decoded)
        
        self.vae = Model(x, x_decoded_mean)
        
        self.vae.load_weights(weights)
        
        self.encoder = Model(x, z_mean)        
        
    def encode(self, x_test):
        
        x_test_encoded = self.encoder.predict(x_test, batch_size=self.batch_size)
        return x_test_encoded
    
    def end_to_end(self, x_test):
        
        x_test_end = self.vae.predict(x_test, batch_size=self.batch_size)
        return x_test_end