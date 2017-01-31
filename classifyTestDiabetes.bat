::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles -train=data/diabetes.train.csv -test=data/diabetes.test.csv -maxBuck=50 -testOut=tmpout/diabetes.test.csv -doOpt=true -doOptClass=1 -optrandomize=false optMaxTime=3
