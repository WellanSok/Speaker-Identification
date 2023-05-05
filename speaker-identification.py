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
from statistics import mode
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
data_dir = '.'  # directory where the data files are stored
data_file = os.path.join(data_dir, "nickTestData1.csv")
data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
data_for_current_speaker[:, -1] = 0
data = np.zeros((0, 8002))
data = np.append(data, data_for_current_speaker, axis=0)
X = np.zeros((0, 984))
y = np.zeros(0, )
for i, window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1]
    # label = data[i][-1]
    label = window_with_timestamp_and_label[-1]
    #if label > 1:
    #    break
    x = feature_extractor.extract_features(window)
    if len(x) != X.shape[1]:
        print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
    X = np.append(X, np.reshape(x, (1, -1)), axis=0)
    y = np.append(y, label)

Y_predict = classifier.predict(X)
print(Y_predict)
YpredictList = list(Y_predict)
print(f"Number of times classifier correctly predicted speaker: {YpredictList.count(0)}")
print(f"Number of Erroneous Predictions: {len(list(filter(lambda x: x != 0,YpredictList)))}")
print(f"Most predicted speaker: {mode(YpredictList)} ")