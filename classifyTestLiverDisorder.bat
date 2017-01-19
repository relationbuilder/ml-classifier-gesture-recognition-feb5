::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles data/liver-disorder.test.csv data/liver-disorder.train.csv 7 > classifyTestLiverDisorder.out.txt
