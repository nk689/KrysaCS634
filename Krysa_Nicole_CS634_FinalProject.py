import warnings 
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
#pip install tensorflow
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')

import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
#pip install tabulate

from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

#RandomForest
from sklearn.ensemble import RandomForestClassifier

#Guassian Naive Bayes
from sklearn.naive_bayes import GaussianNB

#diab = pd.read_csv("diabetes.csv")
print("\n")

#totals of each method
RFTotal = [[0, 0],[0, 0]]
GNBTotal = [[0, 0],[0, 0]]
LSTMTotal = [[0, 0],[0, 0]]

# interpret the confusion matrix
def calculate_results(cm):

  tP = cm[0][0]
  tN = cm[1][1]
  fP = cm[1][0]
  fN = cm[0][1]
  
  P = tP + fN
  N = tN + fP
  
  #print("True positives: " + str(tP))
  #print("True negatives: " + str(tN))
  #print("False positives: " + str(fP))
  #print("False negatives: " + str(fN))

  sensitivity = tP / P
  specificity = tN / N
  precision = tP / (tP + fP)
  accuracy = (tP + tN) / (P + N)
  falsePositiveRate = fP / N
  falseNegativeRate = fN / P
  F1Measure = (2 * tP) / (2 * tP * fP * fN)
  errorRate = (fP + fN) / (P + N)

  print(tabulate([['Accuracy', str(round(accuracy*100, 2)) + "%"], 
                  ['Sensitivity (Recall)', str(round(sensitivity*100, 2)) + "%"], 
                  ['Specificity', str(round(specificity*100, 2)) + "%"], 
                  ['Precision', str(round(precision*100, 2)) + "%"], 
                  ['False Positive Rate', str(round(falsePositiveRate*100, 2)) + "%"], 
                  ['False Negative Rate', str(round(falseNegativeRate*100, 2)) + "%"], 
                  ['Error Rate', str(round(errorRate*100, 2)) + "%\n"]], 
                 headers = ['Measurement', 'Percent'], tablefmt ='orgtbl'))
  print("\n")


# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 

# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets 

#K-Fold 
numSplits = 10
kf = KFold(n_splits = numSplits, shuffle = True, random_state = 42)

for i, (trainIndex, testIndex) in enumerate(kf.split(X), start = 1):
  # Splitting the data
  XTrain, XTest = X.iloc[trainIndex], X.iloc[testIndex]
  yTrain, yTest = y.iloc[trainIndex], y.iloc[testIndex]
  
  #Random Forest
  modelRF = RandomForestClassifier(n_estimators=10, max_features = "sqrt")
  modelRF = modelRF.fit(XTrain, yTrain.values.ravel())
  yPredictionsRF = modelRF.predict(XTest)

  print("Results of the Random Forest execution run #" + str(i) + "\n")
  RFcm = confusion_matrix(yTest, yPredictionsRF)
  tP = RFcm[0][0]
  tN = RFcm[1][1]
  fP = RFcm[1][0]
  fN = RFcm[0][1]

  RFTotal[0][0] += tP
  RFTotal[1][1] += tN
  RFTotal[1][0] += fP
  RFTotal[0][1] += fN
  calculate_results(RFcm)

  # Gaussian Naive Bayes
  modelGNB = GaussianNB()
  modelGNB = modelGNB.fit(XTrain, yTrain.values.ravel())
  yPredictionsGNB = modelGNB.predict(XTest)

  print("Results of the Gaussian Naive-Bayes execution run #" + str(i) + "\n")
  GNBcm = confusion_matrix(yTest, yPredictionsGNB)
  tP = GNBcm[0][0]
  tN = GNBcm[1][1]
  fP = GNBcm[1][0]
  fN = GNBcm[0][1]

  GNBTotal[0][0] += tP
  GNBTotal[1][1] += tN
  GNBTotal[1][0] += fP
  GNBTotal[0][1] += fN
  calculate_results(GNBcm)

  vectorLength = 32
  topWords = len(XTrain) + len(XTest)
  modelLSTM = Sequential()
  modelLSTM.add(Embedding(topWords, vectorLength))
    # 50 memory units, the more memory units allows a model to learn more complex patterns
  modelLSTM.add(LSTM(50)) 


    # Classification problem
    # Use one dense final output layer with a single neuron after the LSTM layer
    # Sigmoid activation (0, 1) for binary classifications
  modelLSTM.add(Dense(1, activation = "sigmoid")) 
  modelLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
  modelLSTM.fit(XTrain, yTrain, validation_data=(XTest, yTest), epochs = 1, batch_size = 500, verbose = 0)
  scores = modelLSTM.evaluate(XTest, yTest, verbose=0)
     #Flatten results to binary values
  yPredictionsLSTM = (modelLSTM.predict(XTest) > 0.5).astype(int)
  
  print("Results of the LSTM execution run #" + str(i) + "\n")
  LSTMcm = confusion_matrix(yTest, yPredictionsLSTM)
  tP = LSTMcm[0][0]
  tN = LSTMcm[1][1]
  fP = LSTMcm[1][0]
  fN = LSTMcm[0][1]

  LSTMTotal[0][0] += tP
  LSTMTotal[1][1] += tN
  LSTMTotal[1][0] += fP
  LSTMTotal[0][1] += fN
  calculate_results(LSTMcm)

print("Final Results: ")
print("Random Forest")
calculate_results(RFTotal)
print("Gaussian Naive Bayes: ")
calculate_results(GNBTotal)
print("LSTM:")
calculate_results(LSTMTotal)

print("Results of the following data value being classified: ")
print("[0,0,1,23,0,0,0,1,1,1,1,1,0,3,15,3,0,0,2,6,7]")
predictionRF = modelRF.predict(np.array([[0,0,1,23,0,0,0,1,1,1,1,1,0,3,15,3,0,0,2,6,7]]))
predictionGNB = modelGNB.predict(np.array([[0,0,1,23,0,0,0,1,1,1,1,1,0,3,15,3,0,0,2,6,7]]))
predictionLSTM = (modelLSTM.predict(np.array([[0,0,1,23,0,0,0,1,1,1,1,1,0,3,15,3,0,0,2,6,7]]))> 0.5).astype(int)

print("Random Forest: " + str(predictionRF))
print("Gaussian Naive Bayes: " + str(predictionGNB))
print("LSTM: " + str(predictionLSTM))

print("Results of the following data value being classified: ")
print("[1,1,1,40,1,1,1,1,1,1,1,1,0,3,15,3,0,0,9,6,7]")
predictionRF = modelRF.predict(np.array([[1,1,1,40,1,1,1,1,1,1,1,1,0,3,15,3,0,0,9,6,7]]))
predictionGNB = modelGNB.predict(np.array([[1,1,1,40,1,1,1,1,1,1,1,1,0,3,15,3,0,0,9,6,7]]))
predictionLSTM = (modelLSTM.predict(np.array([[1,1,1,40,1,1,1,1,1,1,1,1,0,3,15,3,0,0,9,6,7]]))> 0.5).astype(int)

print("Random Forest: " + str(predictionRF))
print("Gaussian Naive Bayes: " + str(predictionGNB))
print("LSTM: " + str(predictionLSTM))