::go build src/classifyFiles.go
classifyFiles -train=data/breast-cancer-wisconsin.adj.data.train.csv -test=data/breast-cancer-wisconsin.adj.data.test.csv -numBuck=10 -WriteJSON=false  -testOut=tmpout/breast-cancer.test.out.csv -doOpt=false -optrandomize=false -optMaxTime=17 -optClassId=4

:: This is an example of data that is so clean and provides such 
:: good predictive input that we don't really gain anything by 
:: running the optimizer.   With a 97% accuracy for no optimizer 
:: with 95% recall at 95% precision catching cases with cancer
:: it is hard to get better.  The optimizer settings above will
:: sometimes improve the precision for class 4 to 100% but generally
:: at the expense of reducing recall for class 4 and precision for 
:: class  2.