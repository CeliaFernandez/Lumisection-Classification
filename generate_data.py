#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:02:47 2017

@author: celiafernandezmadrazo
"""

"""
This script creates two analysis objects that access to the root data and prepare it for the analysis. 

Both good lumisections and bad lumisections are obtained and stored in a pickle file.
"""


from Analysis import Analysis
from Targets import Targets
from Trial import Trial
import os
import random
import numpy as np
import pickle
import time
import progressbar


#%% Get the files

os.chdir("/Volumes/CeFer/Documentos/CERN/Data/fifth_run/crab_SingleMuonRun_2016G-18Apr2017-v1/170808_214948")

path = '0000/'

#%% Analyze the data

myAnalysis = Analysis(path)

for filename in os.listdir(myAnalysis.path): # dir of root files
        
        if 'AOD' in filename:
            myAnalysis.getData(filename)

print('Generating array...')
myAnalysis.getArray()
print('Sorting array...')
myAnalysis.sortArray()
#print('Creating CSV...')
#myAnalysis.createCSV()

featuresNumber = len(myAnalysis.feature_names)

print('-----Data model constructed-----')


#%% Check the data quality

os.chdir("/Users/celiafernandezmadrazo/Documents/CERN/Code/Run Code")

myTargets = Targets()

myTargets.getTargets(runIni = 0, runFin = 300000)
myTargets.sortTargets()
myTargets.createCSV()

print('-----Target model constructed-----')
print('Checking targets...')

goodLumi = []
badLumi = []


i = 0
with progressbar.ProgressBar(max_value=len(myAnalysis.data)) as bar:
    
    for item in myAnalysis.data:
        
        Id = [item[0], item[1]]
        
        if len(item) == featuresNumber:
        
            if Id in myTargets.good: 
                
                goodLumi.append(np.array(item))
                
            elif Id in myTargets.bad:
                
                badLumi.append(np.array(item))
                
            else:
                
                badLumi.append(np.array(item))
                
        time.sleep(0.1)
        bar.update(i)
        i+=1
        
goodLumi = np.array(goodLumi)
badLumi = np.array(badLumi)

#%% Final samples

print('Total number of samples: ' + str(len(myAnalysis.data)))
print('Total good samples :' + str(len(goodLumi)))
print('Total bad samples :' + str(len(badLumi)))

# Targets
goodTargets = np.linspace(1,1,len(goodLumi)) 
badTargets = np.linspace(0,0,len(badLumi))

feature_names = np.array(myAnalysis.feature_names)

# Delete runId and lumiId from the future training variables
goodLumi = goodLumi[:,range(2,len(goodLumi[0]))]
badLumi = badLumi[:,range(2,len(badLumi[0]))]
feature_names = feature_names[2:] 

# Store the variables
print('Storing the variables...')

with open('Variables/lumisection_crab_SingleMuonRun_2016G-18Apr2017-v1.pickle', 'wb') as f:
    pickle.dump([goodLumi, badLumi, goodTargets, badTargets, feature_names], f) # pickle file
    
print('-----Variables stored-----')
    
