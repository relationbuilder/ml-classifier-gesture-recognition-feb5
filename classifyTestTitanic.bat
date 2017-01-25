::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles -train=data/titanic.train.csv -test=data/titanic.test.csv -numBuck=25 -testout=tmpout/Titanic.test.out.csv
