#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:14:10 2017

@author: celiafernandezmadrazo
"""

"""
This script is intended to join all lumisections together. The different lumisections are stored 
in different variable files. For being used in the analysis they need to be stored in the same variable.
"""

import os
import pickle
import numpy as np


path = 'Variables/'

i = 0

for filename in os.listdir(path):
    
    
    if 'SingleMuonRun' in filename:
        
        i += 1
        
        with open(path+filename, 'rb') as f:
            sing_goodLumi, sing_badLumi, sing_goodTargets, sing_badTargets, sing_feature_names = pickle.load(f)
            
            
            print('filename :' + filename)
            print('goodLumi length: '+str(len(sing_goodLumi)))
            
            # Add the different files data to one variable
            
            if i == 1:
                
                goodLumi = sing_goodLumi
                badLumi = sing_badLumi
                goodTargets = sing_goodTargets
                badTargets = sing_badTargets
                
            else:
                
                goodLumi = np.concatenate((goodLumi, sing_goodLumi), axis=0)
                badLumi = np.concatenate((badLumi, sing_badLumi), axis=0)
                goodTargets = np.concatenate((goodTargets, sing_goodTargets), axis=0)
                badTargets = np.concatenate((badTargets, sing_badTargets), axis=0)

feature_names = sing_feature_names # All files are supposed to share this attribute

print('Total number of good lumisections: '+ str(len(goodLumi)))
print('Total number of bad lumisections: '+ str(len(badLumi)))
        
with open('Variables/lumisection_variables.pickle', 'wb') as f:
    pickle.dump([goodLumi, badLumi, goodTargets, badTargets, feature_names], f) # pickle file
        