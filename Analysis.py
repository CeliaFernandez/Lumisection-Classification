#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:28:18 2017

@author: celiafernandezmadrazo
"""

"""
Insert description here
"""

import ROOT
import numpy as np
import os
import pandas as pd

class Analysis:
    
    
    def __init__(self, path_name):
        
        self.data = []
        self.path = path_name
        
        filenames = []    
        for filename in os.listdir(self.path): # dir of root files
            filenames.append(filename)
            
        self.files = filenames        
        
        
    def getData(self, filename):

    # Gets the data stored in the root file whose name is filename
    
        f = ROOT.TFile.Open(self.path + filename)        
        MyAnalysis = f.Get("MyAnalysis")        
        Tree = MyAnalysis.Get("MyTree")
        
#        featureFile = open('features.txt', 'w')
#        featureList = featureFiles.readlines(featureFile)
#        featureList = [n.replace('\n', '') for n in featureList]
#        featureList = featureList[2:]
        muonFeatures = ['qMuPt', 'qMuEta', 'qMuPhi', 'qMuEn_', 'qNVtx', 'qMuCh_'] # muon features that will be added to the analysis (not cosmic muons)
        
        for lumi in Tree:
        
            event = []
        
            event.append(lumi.runId)
            event.append(lumi.lumiId)
            for feature in muonFeatures:
                
                for q in eval('lumi.'+feature):
                    
                    event.append(q)
            
            self.data.append(np.array(event))
            
        self.feature_names = ['runId', 'lumiId', 
                              'qMuPt_mean', 'qMuPt_rms', 'qMuPt_q1', 'qMuPt_q2', 'qMuPt_q3', 'qMuPt_q4', 'qMuPt_q5',
                              'qMuEta_mean', 'qMuEta_rms', 'qMuEta_q1', 'qMuEta_q2', 'qMuEta_q3', 'qMuEta_q4', 'qMuEta_q5',
                              'qMuPhi_mean', 'qMuPhi_rms', 'qMuPhi_q1', 'qMuPhi_q2', 'qMuPhi_q3', 'qMuPhi_q4', 'qMuPhi_q5',
                              'qMuEn_mean', 'qMuEn_rms', 'qMuEn_q1', 'qMuEn_q2', 'qMuEn_q3', 'qMuEn_q4', 'qMuEn_q5',
                              'qMuNVtx_mean', 'qMuNVtx_rms', 'qMuNVtx_q1', 'qMuNVtx_q2', 'qMuNVtx_q3', 'qMuNVtx_q4', 'qMuNVtx_q5',
                              'qMuCh_mean', 'qMuCh_rms', 'qMuCh_q1', 'qMuCh_q2', 'qMuCh_q3', 'qMuCh_q4', 'qMuCh_q5']
        
            
    def getArray(self):
        
        # Gets an array from the data
        
        self.data = np.array(self.data)
        
    def sortArray(self):
        
        # Sort the lumiArray by runId, then by lumiId
        
        run = [self.data[i][0] for i in range(0, len(self.data))] # Get the runId values
        lumi = [self.data[i][1] for i in range(0, len(self.data))] # Get te lumiId
        
        ind = np.lexsort((lumi, run)) # Sort by runId, then by LumiId -> Get the indexes

        self.data = self.data[ind]
        
    def createCSV(self):
      
        if not os.path.exists('CSV output'): os.makedirs('CSV output')
        
        df = pd.DataFrame(data = list(self.data), columns=self.feature_names)
        
        df.to_csv('CSV output/data.csv', index=False, encoding='utf-8')

