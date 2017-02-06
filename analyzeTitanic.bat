::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles -train=data/titanic.train.csv -test=data/titanic.test.csv -maxBuck=120 -testout=tmpout/Titanic.test.out.csv -detToStdOut=false  -doPreAnalyze=true -AnalSplitType=1 -AnalClassId=1  -AnalTestPort=100
