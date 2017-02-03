#!/bin/sh
#rm classifyFiles.exe
#go build src/classifyFiles.go
classifyFiles -train=data/titanic.train.csv -test=data/titanic.test.csv -maxBuck=265 -testout=tmpout/Titanic.test.out.csv
