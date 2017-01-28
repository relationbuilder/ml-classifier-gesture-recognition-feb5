# stock-prep-sma.py
# Convert simple time series stock data into something
# we can possibly use for machine learning classification
# 
# Convert a stock bar file into a tagged stock 
# data more interesting for machine learning.
# The rule will be that the price from a given Bar
# measured as day open must rise by X% before it 
# drops by Y%.  If it does then that bar is classified
# as 1.  Those that fall by more than X% are classified
# as a 0.  
#
# Four our indicators we are going to use a series of
# SMA and compute the slope of the line from the SMA
# for current bar to sma at some bars in the past. 


def sma(vect, numAvgBar):
  rowCnt = 0
  divNum = 0
  runTot = 0
  tout = []
  for anum in vect:
    if (rowCnt > numAvgBar):
      runTot -= vect[rowCnt - numAvgBar]
    rowCnt += 1
    divNum = min(rowCnt, numAvgBar)
    runTot += anum
    avg = runTot / divNum
    tout.append(avg)
  return tout
  
    
    
   




def loadData(fiName):
   oClose = []
   oLow   = []
   oHigh  = []
   with open(fiName, 'r') as fi:
     rows = fi.readlines()[1:]
     for aline in rows:
      row = aline.split(",")
      close = float(row[4])
      low   = float(row[3])
      high  = float(row[2])
      oClose.append(close)
      oLow.append(low)
      oHigh.append(high)
     return  oClose, oHigh, oLow
     


def slope(currNdx, vect, barsPast):
  begNdx = max(0, currNdx - barsPast)
  currVal = vect[currNdx]
  oldVal  = vect[begNdx]
  dif = currVal - oldVal
  slope = (dif / oldVal) / barsPast
  return slope
  
# If we had purchased on this bar would we have made money
# before our stop loss exited the trade.
# return true if price rises by at least goalRisep as portion
# of current close before it drops below goaldropp as portion
# of current close.   Otherwise return 0  
def findClass(currNdx, goalRisep, goalDropp, oClose, oHigh, oLow):
  maxNdx = len(oClose)
  cout = 0
  currClose = oClose[currNdx]
  maxPrice  = currClose + (currClose * goalRisep)
  minPrice  = currClose - (currClose * goalDropp)
  for ndx in range(currNdx+1, maxNdx):
    if oHigh[ndx] > maxPrice:
      return 1  # sucess 
    if oLow[ndx] < minPrice:
      return 0 # failed 
  
  return 0  

def saveData(fiName):
  pass

inName = "data/spy.csv"
print("inName=", inName)
oClose, oHigh, oLow = loadData(inName)
smaLen = 30
portSetTrain = 0.90
print("smaLen=", smaLen)
sma1 = sma(oClose, smaLen)
numBar = len(oClose)
newExt = ".slp" + str(smaLen) + ".csv"
outName = inName.replace(".csv", newExt )
outTrainName = outName.replace(".csv", ".train.csv")
outTestName  = outName.replace(".csv", ".test.csv")

numTrainRow = int((numBar - smaLen) *  portSetTrain)
print("portion of set for Training=", portSetTrain, " #trainRows=", numTrainRow)
print ("trainName=", outTrainName, " testName=", outTestName)

def savePortSet(fiName, begNdx, endNdx):
  with open(fiName, "w") as fout:
    fout.write("class,sl3,sl6,sl12,sl20,sl30,sl60,sl90\n")
    for ndx in range(begNdx,endNdx):
  
      slope1 = slope(ndx,oClose,3)
      slope2 = slope(ndx,oClose,6)
      slope3 = slope(ndx,oClose,12)
      slope4 = slope(ndx,oClose,20)
      slope5 = slope(ndx,oClose,30)
      slope6 = slope(ndx,oClose,60)
      slope7 = slope(ndx,oClose,90)
      bclass = findClass(ndx, 0.01, 0.01, oClose, oHigh, oLow)
      tout = [str(bclass),str(slope1),str(slope2),str(slope3),str(slope4),str(slope5),str(slope6), str(slope7)]
      ts = ",".join(tout)
      print("ndx=",ndx, "s=", ts)
      fout.write(ts)
      fout.write("\n")
  
savePortSet(outTrainName, smaLen+1, smaLen + numTrainRow)
savePortSet(outTestName,  smaLen+numTrainRow+1, numBar)




