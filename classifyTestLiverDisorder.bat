::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles data/liver-disorder.test.csv data/liver-disorder.train.csv 10 > classifyTestLiverDisorder.out.txt
