# ML Quantized classifier in GO#

A general purpose, high performance machine learning
classifier.  Tests are:

   *  ASL Sign language Gesture recognition 
   *  Classify breast cancer from 9 features. 96% accuracy
      at 100% recall first pass with no optimization. Using
      623 training rows.      
   *  Predict death or survival of Titanic passengers. 
   *  Predict Diabetes
   *  Please send me data sets you would like to add 
      to the test.
      
**We Offer Consulting services see: http://BayesAnalytic.com/contact**
      
The library includes TensorFlow Deep Learning implementation of 
classifiers using the same data to compare the run-time 
performance combined with classification recall and accuracy.  

### Quantized Classifier ###
The design for Quantized classifiers was inspired by 
design elements in KNN, Bayesian and SVM engines. 
A key design goal was a faster mechanism to identify 
similarity for a given feature while providing very fast
classification using moderate hardware resources. 

  In KNN we find similar records for a given feature by finding 
  those with the most similar value.  This works but consumes
  a lot of space and run-time.   In Quantized approach 
  we look at the range of data and attempt to 
  group data based on similar values.  EG: if a given
  feature has a value from 0.0 to 1.0 then a 10 bucket system 
  could consider all records that have a value from 0 to 0.1 as 
  similar. Those from 0.1 to 0.2 are similar, etc.  Rather than 
  keeping all the training records we only need to keep 
  the statistics for have many of the records in a given
  bucket belong to 
  each class which we can then use to compute a base probability
  by feature by bucket by class. Applying this across active 
  features gives us a set of probabilities that can be combined
  using ensemble techniques into the probability a given
  row would belong to any of the classes.  
  
  Quantizing the data allows a small memory foot print 
  with fast training without the
  need to keep all the training records in memory. Retaining
  only the statistics allows 
  very large training sets with moderate memory use. 
  The trade off is loosing some of 
  KNN ability to adjust the number of closest neighbors considered 
  quick at runtime.   The offset is that training is so fast that
  the quanta size can be adjusted quickly.  The memory use is so 
  much smaller that we can afford to keep
  multiple models with different quanta sizes loaded and updated
  simultaneously. 


###ASP (American Sign Language) Gesture classifier###
This engine started as a classifier designed to classify Static Gestures for VR with the idea we may be able to produce a useful tool for classifying  ASL using VR input devices.  That is still a primary focus but the core algorithms can be more broadly applied.

See **Overview.pdf** in this repository for conceptual overview of
the approach when using this kind of classifier for gesture recognition.

This repository includes code written to test ideas for static gesture recognition. 

It also includes samples of the classifiers in python that cope
well with smaller training data sets and demonstrate using 
the Quantized classifier approach.  They also handle
massive training data sets with minimal memory.    



 * Version: 0.1
 * License: (MIT) We do sell consulting services http://BayesAnalytic.com/contact
 * Dependencies: 
    - GO Code is cross platform and will run Linux.  This softwar was built using version 1.7.3 windows/amd 64
    - Python code: Was tested with Python 3.5.2 64 bit
    - TensorFlow: Lots of crazy dependencies See: tlearn/tensflowReadme.docx 

### How to Use ###
  * **python quant_filt.py** - Runs test on gesture classification data.
    Shows how quantized concept can be used to implement a
    splay like search tress.  The more quant buckets used 
    the more precise.  This is an alternative to the probability
    model and can provide superior results in some instance.
  
  * **python quant_prob.py** - Runs a test on gesture classification data
    demonstrates quantized probability theory in smallest possible 
    piece of python code.  A more complete version is implemented 
    in classify.go 
    
  * **makeGO.bat** - if you have GO set up then open a command line at
    the base directory containing makeGO.bat and run it.   It should
    setup GOHOME and build the executable files. Tested on windows 10.
    
  * **splitData.bat** - Creates sub .train.csv and test.csv files for the files
    used in the GO classifier tests. Uses splitCSVFile.exe which is built
    by makeGo. 
    
  * **setGoEvn.bat** - will set the GOHOME directory to current working directory
    in a command prompt.
    
  * **go build src/classifyTest.go**
    builds executable classifyTest from GO source. 
    
  * **classifyFiles data/breast-cancer-wisconsin.adj.data.train.csv 
    data/breast-cancer-wisconsin.adj.data.test.csv 10**
    will run the GO based classifier built in GO using
    the first named file for training and the second named
    file for testing will print out results of how well classification
    matches actual source data class.
    
  * **classifyFiles data/titanic.train.csv data/titanic.test.csv 6**
    will run the GO based classifier against the two input files
    this test attempts to predict mortality and will print out
    quality of predictions from classifier compared to known
    result. 
    

### Basic Contents ###
#### GO Based Classifier ####
  src/classifyTest.go
  
  src/csvInfoTest.go
  
  src/splitCSVFile.go 
  
  src/qprob/classify.go
  
  src/qprob/csvInfo.go
  
  src/qprob/util.go
  
  
  
 
  
#### Idea Test Sample Code ####
* **quant_filt.py**  - Machine learning Quantized filter classifier.  This system can provide  
   fast classification with moderate memory use and is easy to see how likely the match is to
   be accurate.

* **quant_prob.py** - Machine learning Quantized probability classifier. Not quite as precise under
   some conditions and quant_filt.py but it can cope with greater amounts of training noise while
   still delivering good results with moderate amounts of training data.  
 

####DATA FILES####
 * **data/data-sources.txt** - Explains sources for the included data files
   some data files are not included and will have to be donwloaded from
   those sources if the usage license was unclear or restrictive.
   
 * **data/train/gest_train_ratio2.csv** - Input training data used for these tests.  We need thousands additional training samples feel free to volunteer after your read overview.pdf in this repository.


####TensorFlow###
 One of the goals this project is to test some
 capabilities of tlearn and TensorFlow using the 
 same data sets.   The assertion is that the 
 tensorflow approach should run faster and require
 less code while producing higher quality classification
 results than my quantized classifier. 
 
* **tlearn/tensFlowReadme.docx** - Notes I made while getting tensor flow running on my windows laptop.


* **tlearn/simple_gestures.py** - sample of reading CSV to  train TensorFlow Model.
   Unfortunately this program while it runs does a pour job of classification. I think
   this is the result of insufficient training data but there is a chance that I still have
   a bug in the interface to TensorFlow.




### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies - Tested on Python 3.5 on Windows 10.
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner Joseph Ellsworth
* We do sell consulting services http://BayesAnalytic.com/contact


#TODO#

##Actions for Both quant_prob and quant_filt##
* Update QuantProb to properly scale buckets to cope with outliers
  to prevent them from negatively affecting spread for normal
  distribution items.

* Add recall, precision by class in classifyFiles

* Implement the TLearn / Tensorflow equivelant of
  classify files.  It could be named CNNClassifyFiles
  since the first implementation would use a convoluted
  neural network.

* Add QProb optimizer that is allowed to change Feature
  weight and feature number of buckets seeking to maximize
  precision at 100% recall. Where total number of buckets
  is considered primary cost.  Minimum number of buckets is
  equal to 2.  a Feature can be turned off by setting it'same
  feaure weight to 0. 

* Add Test set for Daily stock bar data.  Where we add a 
  column which is a SMA(x) where X defaults to 30 days.
  and features are the slope of the SMA(X) when comparing
  current bar to SMA(x) at some number of days in past
  create.  In a stock scenario  you would have a goal EG:
  A Bar for a symbol where price rose by P% within N days 
  without dropping by more than Q% before it reached P%.  
  Those that meet the rule get a class of 1 while those 
  that fail get a class of 0.   
  
  
* Produce GO version of the Quant Filter to see if we can improve
  performance on the diabetes and titanic data set.  The Quant filter 
  is unlikely to deliver 100% recall since it aborts the match as it
  traverses the features when it fails to find a matching bucket id.
  This gives it some precision filtering capability similar to the 
  multi-layer convoluted NN but at a lower cost.  We may be able to 
  add probability since it is still computed by class by feaure. 
  There some chance that more than one class will survive all layers
  of the filter which would mean we need to add a probability to that
  output to act as a tie-breaker. 
  - Add optimizer to quant filter that allows it to 
    vary the number of buckets by feature.  Seting 
    number of buckets to 1 essentially turns a feature
    off by forcing all items into the same bucket. The 
    primary cost function is total number of buckets.  
    The goal of optimizer is based on starting with all
    buckets equal to X.  As it varies the number of buckets
    by feature it can keep the change provided it can 
    increase precision as long as recall does not
    decrease. Or it can increase recall as long as precision does
    not decrease recall.  We can discourage over learning by always
    trying a smaller number of buckets first in the optimizer. 
    A natural side effect of this is that the engine can turn off
    features that do not contribute either accuracy or recall.
    get. 
    

  * Modify Quant_prob run as server handler. 
  * Method will use main as data set name unless &dset is specified.
  * Each named data set is unique and will not conflict with others.
  * Method to add to training data set with POST BODY
  * Method to add to training data set with URI to fetch.
    * Allow the system to skip every N records to reserve for 
    * testing.
  * Method to classify values in file at URI 
    * Allow &testEvery=X to only test every Nth
      item.  This is to support testing.     
  * Method to classify with POST multiple lines.
  * Method to classify with GET for single set of features.
  * Allow number of buckets to be set by column name
  * allow column name to be set map direct to bucket id

  *    
* Produce a version for text parsing that computes position
    indexed position of all words where each unique word gets 
    a column number.   Then when building quantized tree 
    lookup of the indexed position for that word  treat the word 
    index as the bucketId or possibly as columnNumber need to think
    that one through buck as a bucket id seems to make most sense
    nd then 
    treat all the other features as empty. So the list of cols
    may grow to several million but will only contain the hashed
    classes for those value. Allow system to pass in a list
    of columns n the CSV to index as text.  This would not 
    effectively use word ordering but we could use quantized buckets
    for probability of any word being that text in text string so
    a word like "the" that may occur 50 times would occur in a different
    bucket when it is repeated many times. 
  * Only include detail probs if requested.
  * Choose column to use as class



* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)
