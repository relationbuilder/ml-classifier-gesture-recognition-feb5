:: This classify Test requires data that must be 
:: downloaded from a separate site and then converted
:: into our CSV format.  
:: See: convert-cifar-10-to-csv.py for download 
:: and conversion instructions. 
::
:: NOTE: The training input is over 500 megs so it 
::  takes this one a few minutes to run. 
::
classifyFiles data/cifar-10.train.csv data/cifar-10.test.csv 255 > classifyTestCifar10.out.txt
