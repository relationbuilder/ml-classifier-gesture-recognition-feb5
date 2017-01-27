::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles -train=data/liver-disorder.test.csv -test=data/liver-disorder.train.csv -numbuck=10 -testout=tmpout/liver-disorder.test.csv -doOpt=1 -optrandomize=true optMaxTime=10.5

