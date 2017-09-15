#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:46:17 2017

@author: celiafernandezmadrazo
"""

import os
import shutil

os.chdir('/Users/celiafernandezmadrazo/Documents/CERN/Code')

nTest = input("Numero de Test: ")
nEpochs = input("Numero de epochs: ")
nBatch = input("Numero de Batch: ")
nNorm = input("Numero de Norm: ")
nTrain = input("Train on: ")
nVal = input("Validate on: ")

if not os.path.exists('Results/New VAE/Test '+str(nTest)): os.makedirs('Results/New VAE/Test '+str(nTest))

shutil.copy("Run Code/Models/weights.h5", 'Results/New VAE/Test '+str(nTest)+'/')
shutil.copy("Run Code/Histograms/latent_space.png", 'Results/New VAE/Test '+str(nTest)+'/latent_space.png')
shutil.copy("Run Code/Histograms/pt_distribution.png", 'Results/New VAE/Test '+str(nTest)+'/pt_distribution.png')
shutil.copy("Run Code/Training/training_options.pickle", 'Results/New VAE/Test '+str(nTest)+'/training_options.pickle')
shutil.copy("Run Code/Training/loss.png", 'Results/New VAE/Test '+str(nTest)+'/loss.png')
shutil.copy("Run Code/Training/training_variables.pickle", 'Results/New VAE/Test '+str(nTest)+'/training_variables.pickle')
shutil.copy("Run Code/Training/loss.pickle", 'Results/New VAE/Test '+str(nTest)+'/loss.pickle')

infoFile = open('Results/New VAE/Test '+str(nTest)+'/info.txt', 'w')
infoFile.write('TRIAL INFO:\n')
infoFile.write('\n')
infoFile.write('Epochs:\t'+ str(nEpochs)+'\n')
infoFile.write('Batch size:\t'+ str(nBatch)+'\n')
infoFile.write('Norm:\t'+ str(nNorm)+'\n')
infoFile.write('Trained on:\t'+ str(nTrain)+'\n')
infoFile.write('Validated on:\t'+ str(nVal)+'\n')
infoFile.close()