from __future__ import print_function

import numpy as np
import tflearn


# In TFLearn Labels Must be consecutive 
# numbered integers.   This is a general 
# purpose method to map them assigning the
# next unqiue integer to each unique value
def getNumClass(fiName, colNum):
  f = open(fiName)
  lines = f.readlines()[1:]
  f.close()
  maxClass = -989999999
  labVals = []
  for line in lines:
    flds = line.split(",")
    try:
      aclass = int(flds[colNum])
      if aclass >= maxClass:
        maxClass = aclass + 1
    except ValueError:
      aclass = -99999
    labVals.append(aclass)
  return maxClass, labVals

# Locate the best anser in the prediction
# array and return the classId and computed
# prob for that answer. classId start
def best(arr):
  bestP = 0
  bestNdx = -1
  currNdx = 0
  for val in arr:
    if val > bestP:
      bestP = val
      bestNdx = currNdx
    currNdx += 1
  return bestP, bestNdx
    
   
 
def load(fiName):
  numClass, labVal = getNumClass(fiName, 0)
  print("numClass=", numClass)
  # Load CSV file, indicate that the first column represents labels
  from tflearn.data_utils import load_csv
  data, labels = load_csv(fiName,
      target_column=0, categorical_labels=True, n_classes=numClass)

  # Map Convert the number string input into
  # numerics.
  numRow = len(data)
  numCol = len(data[0])
  for rndx in range(0, numRow):
    for cndx in range(0, numCol):
      try:
        data[rndx][cndx] = float(data[rndx][cndx])
      except ValueError:
         data[rndx][cndx] = -9999.0
      
  data = np.array(data, dtype=np.float32)

  #print("data as float array=", data)
  #print("labels=", labels)
  numCol = len(data[0])
  numRow = len(data)
  return (numRow, numCol, numClass, data, labels, labVal)
  

def run(trainFiName, testFiName):
  numRow, numCol, numClass, data, labels, labVal = load(trainFiName)
  tstNumRow, tstNumCol, tstNumClass, tstData, tstLabels, tstLabVal = load(testFiName)

  print("numCol=",numCol, "numRow=", numRow, " numClass=", numClass)

  # Build neural network
  net = tflearn.input_data(shape=[None, numCol])
  net = tflearn.fully_connected(net, 32)
  net = tflearn.fully_connected(net, 32)
  net = tflearn.fully_connected(net, numClass, activation='softmax')
  net = tflearn.regression(net)

  # Define model
  model = tflearn.DNN(net)
  # Start training (apply gradient descent algorithm)
  model.fit(data, labels, n_epoch=6, batch_size=4, show_metric=True)

  
  # Run the prediction for the 
  # Test data set  
  pred = model.predict(tstData)
  rowndx = 0
  sucessCnt = 0
  for rowp in pred:
    bestP, classId = best(rowp)
    actPred = tstLabVal[rowndx]
    labelStr = "fail"
    if actPred == classId:
      labelStr = "success"
      sucessCnt += 1      
    print("class=", classId, " prob=", bestP, " ", labelStr)
    rowndx += 1
  sucPerc = (sucessCnt / tstNumRow) * 100
  failPerc = 100.0 - sucPerc
  print("tested=", tstNumRow, "suceed=", sucessCnt, "good%=", sucPerc, 
        " fail=", tstNumRow - sucessCnt, " failPerc=", failPerc)
    

    
#run('../data/breast-cancer-wisconsin.adj.data.train.csv', '../data/breast-cancer-wisconsin.adj.data.test.csv')

#run('../data/diabetes.train.csv', '../data/diabetes.test.csv')

#run('../data/liver-disorder.train.csv', '../data/liver-disorder.test.csv' )

run('../data/titanic.test.csv', '../data/titanic.test.csv')


