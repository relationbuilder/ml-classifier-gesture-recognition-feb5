set XXCWD=%cd%
cd ..\..\..\..\

classifyFiles -train=data/spy.slp30.train.csv -test=data/spy.slp30.test.csv -maxBuck=60 -testOut=tmpout/spy.slp30.out.csv  -detToStdOut=false -doPreAnalyze=true -AnalSplitType=2 -AnalClassId=1  -AnalTestPort=100

cd %XXCWD%