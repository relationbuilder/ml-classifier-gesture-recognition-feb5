::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles -train=data/diabetes.train.csv -test=data/diabetes.test.csv -numBuck=30 -testOut=tmpout/diabetes.test.csv -doOpt=true -optrandomize=true optMaxTime=3
