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

set XXCWD=%cd%
cd ..\..\..\..\
classifyFiles -train=data/spy.slp30.train.csv -class=data/spy.slp30.class.csv -maxBuck=140 -classOut=tmpout/spy.slp30.pred.class.csv -testOut=tmpout/spy.slp30.out.csv -LoadSavedAnal=true 
cd %XXCWD%

