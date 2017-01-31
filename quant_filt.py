""" Simple gesture classification using quantized filter. 

    Tested with Python 3.52
    
    WARN: Example Only intended to demonstrate principals
    of quantized filter algorithm.  Production version will be
    implemented in src/qprob as a GO library.

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
    
    The system builds the underlying data structure 
    maxBuckets from 2 to GBLMaxNumBuck -1 and returns 
    the list of matches for each level of max buckets.  
    In general a smaller number of buckets will increase 
    recall while a larger number of buckets will increase 
    precision.  If you get a match with a relatively high 
    number of buckets such as 8 or 9 then it is a pretty 
    high quality confidence. 

    Tree[bucketId][bucketId][bucketId][bucketId]... = { containing counts by class }
    If you get a result returning more than one class then the highest probability
    match is for the class with the highest count assuming random distribution of noise
    in the training data.
    

  TODO:  The BuckId calculation is incorrect
   because we need to know the effective range 
   of data values before we can compute a bucket
   size.   The version here will work but it requires
   all values to be converted into a value between
   zero and 1 and even then it would not effectively
   handle outliers.    
 
 """
  

import csv
import sys
import json

class QuantTreeFilt:
  def __init__(self, classCol, maxNumBuck):
    #self.colVals = []
    self.trees =  self.trees = [dict() for x in range(maxNumBuck)]
    self.numCol = 0 # read as part of csv head
    #self.numClass = 0
    self.rowCnt = 0 # number of rows read in training set
    self.maxNumBuck = maxNumBuck # Increase num buckets for max precision reduce for max recall
    self.classCol = classCol
    self.maxByCol = [] # list of max values read by column
    self.minByCol = [] # list of min values read by column
    self.absRange = [] # list of abs range for values in column
    self.stepSizeByCol = [] # list of step sizes computed by col
    self.header = []  # List of field names read from CSV
    
    self.minNumBuck = 2 # Set this to at least 2 if you don't
                        # want a wild guess on some rows.  When
                        # set to 1 it will force a answer for
                        # every row.
                      

  def getBuckId(self, colNum, numBuck, sval):
      
     dval = 0
     try:
       dval = float(sval)
     except ValueError:
       # If can not convert to a float number
       # then just return the input value this
       # allows string values to be indexed as
       # unique buckets
       return sval

    
     if numBuck == 1:
       return 0

      
     absRange = self.absRange[colNum]
     stepSize =  absRange / (numBuck)
     minval = self.minByCol[colNum]
     if dval == minval:
       return 0
     amtOverMin = dval - minval
     numStep = amtOverMin / stepSize
     buckId = int(numStep)
     if numBuck == 1 and buckId != 0:
       print("L72: colNum=", colNum, "dval=", dval, " sval=", sval, "minVal=", minval,  "numBuck=", numBuck,"stepSize=", stepSize, "amtOverMin=",
            amtOverMin, " numStep=", numStep, " absRange=", absRange, " buckId=", buckId)
     # Note: BuckId will be negative when dval
     # is less than minValue encountered during
     # training
     return buckId
     
    
  # Updates stats in tree form from
  # 1 bucket through self.maxNumBuck
  def updateStats(self, arow, numBuck):        
    classCol = self.classCol
    trees = self.trees
    rclass = int(arow[classCol])    
    
    # Initialize our current buck to
    # starting point in our tree.
    currBuck = trees[numBuck]

    # Build a nested Tree containing
    # one layer per feature
    for ndx, colVal in enumerate(arow, start=0): 
      if ndx != classCol:        
        buckId = self.getBuckId(ndx, numBuck, colVal)
        if not buckId in currBuck:
          # If bucket does not exist then
          # we need a new one to track this value
          #print("L100: Create new Bucket id=", buckId)
          currBuck[buckId] = {}  
        currBuck = currBuck[buckId]  
        #print("L103: buckId=", buckId, "rclass=", rclass, "currBuck=", currBuck)
        
    # Finished walking the bucket tree 
    # to reach sentinal value for the number
    # of Columns Now record count by class  
    if not rclass in currBuck:
      currBuck[rclass] = 1
    else:
      currBuck[rclass] += 1
    # Keep a Total Count for all classes 
    # at this level so we can compute a 
    # Base probability
    if not "t" in currBuck:
      currBuck["t"] = 1
    else:
      currBuck["t"] += 1   
      
  def readTrainingData(self, fiName):                
    with open(fiName, 'r') as csvfile:
      rreader = csv.reader(csvfile, dialect='excel', delimiter=',', quotechar='"')
      rowCnt = 0
      for row in rreader:    
        if rowCnt == 0:
          header = row
        else:
          for nb in range(1, self.maxNumBuck):
            self.updateStats(row, nb)
        rowCnt += 1

        
  def readMinMax(self, fiName):
    classCol = self.classCol
    maxNumBuck = self.maxNumBuck   
    numCol = self.numCol
    
    with open(fiName, 'r') as csvfile:
      rreader = csv.reader(csvfile, dialect='excel', delimiter=',', quotechar='"')
      rowCnt = 0
      numCol = 0
      for row in rreader:    
        if rowCnt == 0:
          # Reading the header row so we initialize
          # a bunch of instance variables based on
          # how many features we detected. 
          self.header = row
          numCol = len(row)
          self.numCol = numCol
          # Initialize array to hold min/max
          # values 
          self.maxByCol =  [-99999999999.0] * numCol
          self.minByCol =  [999999999999.0] * numCol
          self.absRange = [0] * numCol
          self.stepSizeByCol = [0] * numCol
          self.colVals = [dict() for x in range(numCol)]
         
                 
        else:
          # Reading a real row 
          for nb in range(1, maxNumBuck):
            for ndx, dval in enumerate(row, start=0):
              fval = 0
              try:
                fval = float(dval)
              except ValueError:
                continue
              if fval > self.maxByCol[ndx]:
                self.maxByCol[ndx] = fval
              elif fval < self.minByCol[ndx]:
                self.minByCol[ndx] = fval                        
        rowCnt += 1
      self.numRow   = rowCnt

      # Now compute the actual abs range by column
      # used to compute step size latter. 
      for ndx, maxVal in enumerate(self.maxByCol , start=0):        
        self.absRange[ndx] = maxVal - self.minByCol[ndx]

      print("L192: self.minByCol=",self.minByCol)
      print("L193: self.maxByCol=",self.maxByCol)
      print("L194: self.absRange=", self.absRange)
      
        
      
  def matchNumBuck(self, drow, numBuck): 
    classCol = self.classCol    
    rclass =  drow[classCol]
    colndx = 0
    currBuck = self.trees[numBuck]
    #print("L189: currBuck=", currBuck)
    for ndx, fval in enumerate(drow, start=0):
      if ndx != classCol:        
        buckId = self.getBuckId(ndx, numBuck, fval)
        #print ("L195 fval=", fval, " buckId=", buckId)
        if not buckId in currBuck:
          #print("L195 fail ndx=", ndx, " fval=", fval, " buckId=", buckId, "numBuck=", numBuck, " currBuck=", currBuck)
          return None
        else:
          currBuck = currBuck[buckId]
          #print ("L199: buckId=", buckId, " currBuck=", currBuck)
    return currBuck # This is the count by classId
         
  # Since the most precise trees will end up returning
  # none when there is not match we compute the probability
  # based on less granular.  This higher the number of 
  # buckets where we find a match the better the quality of our
  # match. 
  def match(self, features):
    tout = {}
    for nb in range(self.minNumBuck, self.maxNumBuck):
      tout[nb] = self.matchNumBuck(features, nb)
    return tout
    

  ## TODO:  Think about a match where we walk the tree 
  ##  using the maximum number of buckets where we find 
  ##  a match for each feature.  If we can not find a match
  ##  For that item then we walk back to the next most 
  ##  reduced item for that feature.  EG: In some features
  ##  We want to use 8 buckets for others we may need to 
  ##  use 2 buckets that means we have to detect failure 
  ##  at a given point and backtrack until we find a match
  ##  but once back tracked we have to walk forward from there

  # Determine quality of match 
  def chooseRowResult(self, rowRes):                            
     currNdx = -999999999
     currRow = None
     for ndx in rowRes:
       #print("L179: ndx=", ndx)
       fval = rowRes[ndx]
       #print ("L179: ndx=", ndx, " fval=", fval)
       if fval != None and ndx > currNdx: 
           # Match with the highest ndx will 
           # be the most specific match 
           currNdx = ndx
           currRow = fval
     #print("L254: currNdx=", currNdx, "currRow=", currRow)
     return currNdx, currRow     
      
  def readTestData(self, fiName):
    classCol = self.classCol
    tout = []
    with open(fiName, 'r') as csvfile:
      rreader = csv.reader(csvfile, dialect='excel', delimiter=',', quotechar='"')
      rowCnt = 0
      for row in rreader:    
        if rowCnt == 0:
          testHeader = row      
        else:
          actClassId = int(row[classCol])
          #print("L262 row=", row)
          # Reading data values
          matchRes = self.match(row)
          #print("L257: matchRes=", matchRes)
          level, choice = self.chooseRowResult(matchRes)
          
          # TODO: Move This section Out to separate Method
          # Interpret results 
          #print("L266: level=", level, " choice=", choice)
          bestClass = None
          bestClassCnt = 0
          if choice == None:
            #print("L279: No choice for row ", rowCnt, " row=", row)
            tout.append({"act" : actClassId, "stat" : "fail",
                         "reason" : "noMatch", 
                         "row=" : row})
            continue            
          totCnt = choice["t"]
          
          # Find our Best Class out of choices
          # at this level and create prob by
          # class for each results
          trec = {}
          trec["tot"] = totCnt
          for cid in choice:
            if cid == "t":
              continue            
            cnt = choice[cid]            
            #print("L281: cid=", cid, " cnt=", cnt, "totCnt=", totCnt)         
            prob = cnt / totCnt
            cObj = { "cnt" : cid, "prob" : prob }
            trec[cid] = cObj            
            if cnt > bestClassCnt:
              bestClassCnt = cnt
              bestClass = cid
          trec["best"] = bestClass
          trec[bestClass]["best"] = True
          trec["act"] = actClassId
          trec["lev"] = level
          #print("L294: trec=", trec)
          
          # Record whether our best match
          # coincides with our actual class
          if bestClass == actClassId:                                    
            trec["stat"] = "ok"
          else:
            trec["stat"] = "fail"                               
          tout.append(trec)
        rowCnt += 1
    return tout

def makeEmptyClassSum(id):
  return {"id" : id, "totCnt" : 0, "sucCnt" : 0, "noClass" : 0,
          "taggedCnt" : 0,  "precis" : 0.0, "recall" : 0.0}


def analyzeTestRes(res):  
  rrecs = []
  totCnt = 0
  sucCnt = 0
  failRateCnt = 0
  byClass = {}
  tout = { "byClass" : byClass, "NoClass" : 0 }
  
  for rrow in res:
    totCnt += 1
    stat = rrow["stat"]
    actClass  = rrow["act"]
        
    if not actClass in byClass:
      byClass[actClass] =  makeEmptyClassSum(actClass)
    byClass[actClass]["totCnt"] += 1

    if not "best" in rrow:
      tout["NoClass"] += 1
      rrecs.append(rrow)
      byClass[actClass]["noClass"] += 1
      continue
    
    cid = rrow["best"]
    if not cid in byClass:
      byClass[cid] =  makeEmptyClassSum(cid)      
    tagClass = byClass[cid]
    tagClass["taggedCnt"] += 1
    
    if stat == "ok":
      sucCnt += 1
      tagClass["sucCnt"] += 1
      
    prob = rrow[cid]["prob"]
    trow = [cid, prob, actClass, stat]
    rrecs.append(trow)

  tout["NumRow"] = totCnt
  tout["NumPred"] = totCnt - tout["NoClass"] 
  prec = sucCnt / tout["NumPred"] 
  tout["SucessCnt"] = sucCnt
  tout["FailCnt"] = tout["NumPred"] - sucCnt 
  tout["Precision"] = prec  
  tout["NoClassRate"] = tout["NoClass"] / totCnt
  tout["TotRecall"] = (totCnt - tout["NoClass"]) / totCnt
  
  for classId in byClass:
    aclass = byClass[classId]
    aclass["fail"] = aclass["taggedCnt"] - aclass["sucCnt"]
    aclass["classProb"] = aclass["totCnt"] / totCnt
    try:
      aclass["precis"] = aclass["sucCnt"] / aclass["taggedCnt"]
    except ZeroDivisionError:
      aclass["precis"] = -1

    try:
      aclass["recall"] = aclass["sucCnt"] / aclass["totCnt"]
    except ZeroDivisionError:
      aclass["recall"]
      
  return tout, rrecs
    
    
    
  
      
def processTest(trainFiName, testFiName, maxNumBuck):
  print("trainFiName=", trainFiName, " testFiName=", testFiName, " maxNumBuck=", maxNumBuck)
  qf =  QuantTreeFilt(0, GBLMaxNumBuck)
  qf.readMinMax(trainFiName)
  qf.readTrainingData(trainFiName)
  
  tout = qf.readTestData(testFiName)
  #print (" tout=", tout)

  analyzedRes, recs = analyzeTestRes(tout)
  
  print("trainFiName=", trainFiName, " testFiName=", testFiName, " maxNumBuck=", maxNumBuck)
  #print ("\n\nAnalyzed recs=", recs)
  print ("\n\n\nAnalyzed\n", json.dumps(analyzedRes, sort_keys=True, indent=3))
   
## ---- 
## -- MAIN
## ----

GBLMaxNumBuck = 10 # Increase num buckets for max precision reduce for max recall

#processTest('data/gest/gest_train_ratio2.csv', 'data/gest/gest_test_ratio2.csv', GBLMaxNumBuck)

#processTest('data/breast-cancer-wisconsin.adj.data.train.csv', 'data/breast-cancer-wisconsin.adj.data.test.csv', 11)

#processTest('data/diabetes.train.csv', 'data/diabetes.test.csv', GBLMaxNumBuck)

#processTest('data/liver-disorder.train.csv', 'data/liver-disorder.test.csv', GBLMaxNumBuck)

#processTest('data/wine.data.usi.train.csv', 'data/wine.data.usi.test.csv', GBLMaxNumBuck)

processTest('data/slv.slp30.train.csv', 'data/slv.slp30.test.csv', 500)

#processTest('data/spy.slp30.train.csv', 'data/spy.slp30.test.csv', 8)

#processTest('BodyStateModel-paper_train.csv', 'BodyStateModel-paper_test.csv', 10)

