# -*- coding: utf-8 -*-
########## Extra Credit ##########################
import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import FeatureExtractor
import os

# Load the classifier:
output_dir = 'training_output'
classifier_filename = 'classifier.pickle'

with open(os.path.join(output_dir, classifier_filename), 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
feature_extractor = FeatureExtractor(debug=False)
    


## Write the code for test code"
