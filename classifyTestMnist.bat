::rm classifyFiles.exe
::go build src/classifyFiles.go
classifyFiles data/mnist.train.csv data/mnist.test.csv 155 > classifyTestMnist.out.txt
