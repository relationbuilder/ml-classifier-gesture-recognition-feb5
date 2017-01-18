# TensorFlow Readme #
 One of the goals this project is to test some
 capabilities of tlearn and TensorFlow using the 
 same data sets.   The assertion is that the 
 tensorflow approach should run faster and require
 less code while producing higher quality classification
 results than my quantized classifier. 
 

These Tensorflow wrappers are intended to allow us to test TensorFlow
against the same set of data we used to test the Quantise probability
and quantized filter classifiers we first demonstrated in Python and
then built production version in GO.   The intent is to measure 
* A) Runtime for training
* B) Runtime for classifing
* C) Memory cost for training
* D) Memory cost for classification
* E) Classification accuracy at 100%
* F) Classification recall at a given accuracy
* G) Test the common aglorithms supported by TensorFlow
     to see which ones perform better.  We will start with
     CNN the Convoluted Neural Networks and add more as I
     have time.
     
## Sample Use ##

## Important Files ##


* **[TensorFlow-and-TFLearn-Readme.html](TensorFlow-and-TFLearn-Readme.html)** - Notes I made while getting tensor flow running on my windows laptop.


* **[tlearn/simple_gestures.py](tlearn/simple_gestures.py)** - sample of reading CSV to  train TensorFlow Model.
   Unfortunately this program while it runs does a pour job of classification. I think
   this is the result of insufficient training data but there is a chance that I still have
   a bug in the interface to TensorFlow.

* **[classify_breast_cancer.py](classify_breast_cancer.py)** - Sample 
  attempting to classify the [breast cancer](../data/breast-cancer-wisconsin.adj.data.csv)
  data files in the to compare against the results from 
  [classifyTestBCancer.bat](../classifyTestBCancer.bat) 
  Tensorflow can not cope with some of the data in the source
  file so we must transform into a version Tensorflow can
  cope with then we can run the training.  After that we 
  must transform the training data set and then run the
  tensorflow classify methods exract the results and output
  them in a form we can read easily. 
  

## Special Requirements  ##
TensorFlow can only cope with very specific shapes 
of data so it is more difficult to build  ClassifyFiles
component like we did for the Quantized probability
classifier we built in GO.    
* All Class Id must integers starting with 0 and rising
  with no gaps.  EG: They can not be 2,4,10 they must be
  0,1,2 if there are 3 discrete classes.


#Actions#

* classify_breast_cancer.py modify file loader to be
  general purpose.
  
* Convert classify_breast_cancer.py into general purpose
  CNNClassifyFiles.py that can handle any file provided 
  the class is in column 1 and all values are int, float
  or can safely be set to 0.0 if they are strings.   Will
  need a more complex transform for files that contain 
  strings but I think I can re-purpose the transform keys
  value if we detect any column can not be transformed
  to float safely. 
  
