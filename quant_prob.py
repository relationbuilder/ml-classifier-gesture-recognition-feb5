""" Simple gesture classification using quantized probability 

    We quantize the feature values 
    into discrete buckets and that are fast to compute and lookup. To match
    we compute the bucket id for each feature then keep a counter for the 
    class ids that were referenced. Larger sample sets give us a larger number
    of buckets to accomodate variations.

  We index all the features so we lookup the set of descriptions that
  are closest into a dictionary by class 
  that we use to load the
  find the closest match for each attribute and compute a distance
  from that class.
  produces colVals[1..NumFeat]
     Each ColVal = {} Indexed by bucketId
       Each classId = {} Indexed RClass 
         colVals[columnNumber][BucketId][classId] = CountOfRowsMatching
 
 """
  

import csv


colVals = []
features = {}
numFeat = 0
numClass = 0
rowCnt = 0 # number of rows read in training set
GBLNumBuck = 10.0


def updateStats(arow):
  rclass = int(arow[0])
  rest = arow[1:]
  colndx = 0;
  for colVal in rest:
    buckId = int(float(colVal) * GBLNumBuck)
    acol = colVals[colndx]
    if not buckId in acol:
     acol[buckId] = {}
    aBuckSet = acol[buckId]    
    if not rclass in aBuckSet:
     aBuckSet[rclass] = 0
    aBuckSet[rclass] += 1
    colndx += 1
    

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
        updateStats(row)
      rowCnt += 1
    


def match(features): 
  colndx = 0
  cnts = {}
  totCnt = 0
  # TODO: To make better statistically we need to 
  #  Compute the probability for each feature being in 
  #  of a given class by feature the combine those 
  #  probabilties.  That way we will give higher priority
  #  to buckets that contain fewer records belonging to
  #  another class.  See Actual Bayesian prob which includes
  #  probability of not being in the class.
  #
  # Update the count for matching bucket
  for fval in features:
    buckId = int(float(fval) * GBLNumBuck)
    acol = colVals[colndx]
    if buckId in acol:
      aBuckSet = acol[buckId]
      for classId in aBuckSet:
        featBuckCnt= aBuckSet[classId]
        totCnt += 1
        if not classId in cnts:
          cnts[classId] = 0
        cnts[classId] += 1
    colndx += 1  
  #print("cnts=", cnts)
  # Convert raw counts in to probability
  prob = {}
  bestClass = None
  bestCnt   = 0
  for classId in cnts:
    acnt = cnts[classId]
    prob[classId] = acnt / totCnt;
    if acnt > bestCnt:
      bestClass = classId
      bestCnt = acnt
  return { 'best' : bestClass, 'counts=' : cnts, 'prob=' : prob }
    
    
## ---- 
## -- MAIN
## ----
readTrainingData('data/train/gest_train_ratio2.csv')
print("colVals=", colVals)

 
matchHSamp =  match([0.080,0.10,0.964,1.000,0.632,0.825])
matchYSamp =  match([0.040,0.100,0.144,0.205,0.750,0.725])

print ("matchHSamp = ", matchHSamp)
print ("mathYSamp = ", matchYSamp)

