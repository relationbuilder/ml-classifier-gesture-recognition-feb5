::rm classifyFiles.exe
::go build src/classifyFiles.go
::
:: Attempts to classify data based on which bars will
:: go up by 1% before they drop by 1%.  With the assumption
:: that greater than 50% accuracy could be profitable 
:: depending on trading fees.   Those that go up are 
:: classified as 1 those that drop are classified as 0
:: The classification step is done by stock-prep-sma.py 
:: which also splits the data between train and test file.
::
:: To Run the Stock example on SPY data first run the 
:: the utility to download data from yahoo python yahoo-stock-download.py
:: then run utility to convert raw stock data into machine learning
::  data python stock-prep-sma.py then  you can run
:: this module.
classifyFiles -train=data/spy.slp30.train.csv -test=data/spy.slp30.test.csv -maxBuck=140 -testOut=tmpout/spy.slp30.out.csv -doOpt=false -optrandomize=false -optMaxTime=1  -OptClassId=1 -OptMaxPrec=0.7 -OptMinRecall=0.001

:: -optMaxTime=19   Number of seconds the optimer is allowed to run
::
:: -doOpt=true      Allows optimizer to run 
::
:: -optClassId=1    Tells optimizer to focus on improving results for class 1%
::                  predicts stock will go up 1% before it drops by 1%
:: 
:: -OptMaxPrec=0.53 Tells optimizer to focus in improving Recall once 
::                  precision exceeds 53%.  It will still accept changes
::                  to improve accuracy but it will also accept changes 
::                  that improve recall provided it does not reduce 
::                  accuracy below 53%.