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

See **[Overview.pdf](Overview.pdf)** in this repository for conceptual overview of
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
  * Install python. We tested 3.5.2 but should work with newer versions.
    only needed if you want to run TensorFlow or Python samples we supplied.
  
  * Install GO. 
  
  * Install TensorFlow, TFLearn and run their basic tests to ensure they
    run correctly.  This may also require installing CUDA depending on 
    whether you want to use the GPU version of TensorFlow.  TFLearn requires
    Python we tested ours with python 3.5.2
  
  * **setGoEvn.bat** - will set the GOHOME directory to current working directory
    in a command prompt.  This is required for the GO compiler to find the
    source code. Tested on windows 10 but should be similar on linux.

  * **[makeGO.bat](makeGO.bat)** - First install GO and ensure it has
    been added to PATH.  Open a command line at
    the base directory containing makeGO.bat and run it. It
    will build the executables based on GO that are needed to run 
    the tests. Tested on windows 10 but should be similar on linux.
    

  * **[splitData.bat](splitData.bat)** - Creates sub .train.csv and 
    test.csv files for the files used in the classifier tests. Uses splitCSVFile.exe which is built by makeGo.  Run this before 
    attempting to run the classifier to ensure proper data is present.

    
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
    
    
  * **python quant_filt.py** - Runs test on gesture classification data.
    Shows how quantized concept can be used to implement a
    splay like search tress.  The more quant buckets used 
    the more precise.  This is an alternative to the probability
    model and can provide superior results in some instance.
  
  * **python [quant_prob.py](quant_prob.py)** - Runs a test on
    gesture classification data demonstrates quantized probability
    theory in smallest possible piece of python code.  A more 
    complete version is implemented in classify.go 
        
### Basic Contents ###
* **[todo.md](todo.md)** - list of actions and enhancements roughly
    prioritized top down.
    
#### GO Based Classifier ####
  
  src/classifyTest.go
  
  src/csvInfoTest.go
  
  src/splitCSVFile.go 
  
  src/qprob/classify.go
  
  src/classifyFiles.go
  
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




* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)
