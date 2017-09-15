#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:24:47 2017

@author: celiafernandezmadrazo
"""

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
import keras

def norm1(x, mu, sigma):
    
    return (x - mu)/sigma 

def norm2(x, mu, maxi, mini):
    
    return (x - mu)/(maxi-mini) 

def denorm1(y, mu, sigma):
    
    return y*sigma + mu

def denorm2(y, mu, maxi, mini):
    
    return y*(maxi-mini) + mu 


def norm3(x, mu, maxi, mini):
    
#    if maxi - mini == 0.0:
#        
#        toreturn = x
#        
#    else:
        
     return (x - mini)/(maxi-mini) 
    
#    return toreturn

def denorm3(y, mu, maxi, mini):
    
#    if maxi - mini == 0.0:
#        
#        toreturn = y
#        
#    else:
        
     return y*(maxi-mini) + mini 
    
#    return toreturn

def norm4(x, mu, maxi, mini):
    
    y = x/maxi
    
    return y-mini/maxi

def denorm4(y, mu, maxi, mini):
    
    x = y + mini/maxi
    
    return x*maxi


