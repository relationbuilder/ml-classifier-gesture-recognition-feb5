::rm classifyFiles.exe
::go build src/classifyFiles.go
::
::
:: To Run the Stock example on SPY data first run the 
:: the utility to download data from yahoo python yahoo-stock-download.py
:: then run utility to convert raw stock data into machine learning
::  data python stock-prep-sma.py then  you can run
:: this module.
classifyFiles -train=data/spy.slp90.train.csv -test=data/spy.slp90.test.csv -maxBuck=10 -testOut=tmpout/spy.slp90.out.csv 

