::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles data/titanic.train.csv data/titanic.test.csv 25 > classifyTestTitanic.out.txt
