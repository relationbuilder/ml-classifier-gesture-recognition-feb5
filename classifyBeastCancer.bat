::go build src/classifyFiles.go
:: WriteJSON=true causes most output to be written
::     in JSON format.
::
:: WriteFullCSV=true  Causes the classifier to generate
::     a new CSV file the same as original except the 
::     class has been changed to the predicted class. 
::
classifyFiles -train=data/breast-cancer-wisconsin.adj.data.train.csv -class=data/breast-cancer-wisconsin.adj.data.class.csv -numBuck=10 -WriteJSON=false -classOut=tmpout/breast-cancer.class.out.csv  -WriteFullCSV=true  -detToStdOut=true

:: Try with -detToStdOut=false to supress some of details being 
::    written to stdout