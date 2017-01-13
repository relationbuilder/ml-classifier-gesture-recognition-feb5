""" Simple gesture classification using quantized filter. 

    Eg: The AT Feature has the following discrete values 
    1.0, 0.79, 0.505, 0.47, 0.70, 0.40, 0.08  We quantize the values 
    into discrete buckets for feature 1.   For each of these buckets we 
    we create list of buckets for feature 2.  For each of these buckets we 
    create a list of buckets for feature 4 until we run out of features. 
    
    This data structure was inspired by splay search trees 
    
    This allows a very fast lookup filter so only those records that have a 
    matching filter all the way down the tree survive.  When we have very large
    training sets with limited noise this can be a very fast and powerful algorithm.
    At the termal bucket for the last feature we keep the list of classes with a counter
    of the number of examples that were placed in that bucket. 
    
    The system builds the underlying data structure maxBuckets from 2 to GBLMaxNumBuck -1
    and returns the list of matches for each level of max buckets.    In general a smaller
    number of buckets will increase recall while a larger number of buckets will 
    increase precision.  If you get a match with a relatively high number of buckets 
    such as 8 or 9 then it is a pretty high quality confidence. 

    Tree[bucketId][bucketId][bucketId][bucketId]... = { containing counts by class }
    If you get a result returning more than one class then the highest probability
    match is for the class with the highest count assuming random distribution of noise
    in the training data.
    
    
 
 """
  

import csv


colVals = []
trees = {}
numFeat = 0
numClass = 0
rowCnt = 0 # number of rows read in training set
GBLMaxNumBuck = 15 # Increase num buckets for max precision reduce for max recall


def updateStats(arow, numBuck):
  rclass = int(arow[0])
  rest = arow[1:]
  colndx = 0;
  if not numBuck in trees:
    trees[numBuck] = {}
  currBuck = trees[numBuck]
  for colVal in rest:
    buckId = int(float(colVal) * numBuck)
    if not buckId in currBuck:
      currBuck[buckId] = {}
    currBuck = currBuck[buckId]
    colndx += 1
  if not rclass in currBuck:
    currBuck[rclass] = 1
  else:
    currBuck[rclass] += 1

    
def readTrainingData(fiName):
  global numFeat, colVals, rowCnt
  with open(fiName, 'r') as csvfile:
    rreader = csv.reader(csvfile, dialect='excel', delimiter=',', quotechar='"')
    rowCnt = 0
    for row in rreader:    
      if rowCnt == 0:
        header = row
        numFeat = len(row)- 1
        colVals = [dict() for x in range(numFeat)]        
      else:
        for nb in range(2, GBLMaxNumBuck):
          updateStats(row, nb)
      rowCnt += 1
    


    
def matchNumBuck(features, numBuck): 
  colndx = 0
  cnts = {}
  totCnt = 0
  currBuck = trees[numBuck]
  for fval in features:
    buckId = int(float(fval) * numBuck)
    #print ("fval=", fval, " buckId=", buckId)
    if not buckId in currBuck: 
      return None
    else:
      currBuck = currBuck[buckId]
    #print ("currBuck=", currBuck)
  return currBuck # This is the count by classId
    
    
# Since the most precise trees will end up returning
# none when there is not match we compute the probability
# based on less granular.  This higher the number of 
# buckets where we find a match the better the quality of our
# match. 
# 
def match(features):
  tout = {}
  for nb in range(2, GBLMaxNumBuck):
    tout[nb] = matchNumBuck(features, nb)
  return tout
## ---- 
## -- MAIN
## ----
readTrainingData('data/train/gest_train_ratio2.csv')
print("Trained data trees=", trees)


matchHSamp =  match([0.080,0.10,0.864,1.000,0.632,0.855])
print ("matchHSamp = ", matchHSamp)

matchYSamp =  match([0.040,0.100,0.144,0.205,0.750,0.725])


print ("mathYSamp = ", matchYSamp)

