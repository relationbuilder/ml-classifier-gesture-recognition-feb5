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
     CNN the Convolutional Neural Networks and add more as I
     have time.

## Sample Use ##

* **[CNNClassifyBCancer.bat](CNNClassifyBCancer.bat )**

* >> Runs the Tensorflow NN classifier on the Breast Cancer data set.  Uses CNNClassify.py Prints out the classification results.   

* **[CNNClassifyDiabetes.bat](CNNClassifyDiabetes.bat)**

* >> Runs the Tensorflow NN classifier using the Diabetes training and test data. Uses [CNNClassify.py](CNNClassify.py) Prints out the classification results. 
  >>
* **[CNNClassifyLiverDisorder.bat](CNNClassifyLiverDisorder.bat)**

  >>Runs the Tensorflow NN classifier using the Diabetes training and test data. Uses [CNNClassify.py](CNNClassify.py) Prints out the classification results. 

* **[CNNClassifyTitanic.bat](CNNClassifyTitanic.bat)
  >>Runs the Tensorflow NN classifier using the Titanic Survivor training and test data. Uses [CNNClassify.py](CNNClassify.py) Prints out the classification results. 
* **[CNNClassifyWine](CNNClassifyWine)
  >> Runs the Tensorflow NN classifier using the Wine Tasting training and test data. Uses [CNNClassify.py](CNNClassify.py) Prints out the classification results.   The Tensorflow configuration only scored 64% with n_epoch equal to 30 but when increased to 90 the score improved to 92%. 
* **python [CNNClassify.py](CNNClassify.py) ../data/breast-cancer-wisconsin.adj.data.train.csv ../data/breast-cancer-wisconsin.adj.data.test.csv 30 ** 

* > > This is done automatically by CNNClassifyBCancer.bat.   Included here to illustrate how to do the same manually. Runs the Tensorflow NN classifier reading the .train.csv file for traing and using the .test.csv file to supply test data. It prints out the classification results.  

## Important Files ##

* **[CNNClassify.py](CNNClassify.py)** Runs the Tensorflow NN classifer reading the .train.csv file for traing and using the .test.csv file to supply test data. It prints out the classification results and shows the precision when forced to 100% recall.  To the best of my knowledge this is the only TensorFlow utility that can run across many different  CSV files that contain differnt sets of clases and different numbers of columns without changing the code.  

* **[TensorFlow-and-TFLearn-Readme.html](TensorFlow-and-TFLearn-Readme.html)** - Notes I made while getting tensor flow running on my windows laptop.


>##Obsolete##
>* **[tlearn/simple_gestures.py](tlearn/simple_gestures.py)** - sample of >reading CSV to  train TensorFlow Model.   Unfortunately this program while it runs does a pour job of classification. I think   this is the result of insufficient training data but there is a chance that I still have a bug in the interface to TensorFlow.

 

# Reference Links

- I discovered this Tensorflow classifer after I had already written my own. His code didn't use the TLearn library so it may have fewer dependencies. [A neural network in TensorFlow for classifying UCI breast cancer data](http://vprusso.github.io/blog/2016/tensor-flow-neural-net-breast-cancer/) by Vincent Russo
- ​

#Actions#
* Add command line parms parser to CNNClassify

* **[CNNclassifyBreastCancer.bat](CNNClassifyBreaskCancer.bat)** - Sample 
  attempting to classify the [breast cancer](../data/breast-cancer-wisconsin.adj.data.csv)
  data files in the to compare against the results from 
  [classifyTestBCancer.bat](../classifyTestBCancer.bat) 
  Tensorflow can not cope with some of the data in the source
  file so we must transform into a version Tensorflow can
  cope with then we can run the training.  After that we 
  must transform the training data set and then run the
  tensorflow classify methods exract the results and output
  them in a form we can read easily. 


# DONE #

* Convert classify_breast_cancer.py into general purpose
  CNNClassifyFiles.py that can handle any file provided 
  the class is in column 1 and all values are int, float
  or can safely be set to 0.0 if they are strings.   Will
  need a more complex transform for files that contain 
  strings but I think I can re-purpose the transform keys
  value if we detect any column can not be transformed
  to float safely. 
