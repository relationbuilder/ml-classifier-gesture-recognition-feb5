go build src/classifyFiles.go
classifyFiles data/breast-cancer-wisconsin.adj.data.train.csv data/breast-cancer-wisconsin.adj.data.test.csv 10 > classifyTestBCanser.out.txt
