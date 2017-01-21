#Future Work List#
**Please coment and critique.**  I need your help to ensure future work is prioritized to help the people actually  using the library.  http://BayesAnalytic.com/contact

##Focus Areas##
There are several focus areas where I can invest time. Please let me know if you have an opinion for prioritizing between them. 

* **Implement the optimizer / discovery aspect.**  I explain some of this in the [Genomic research white paper](genomic-notes.md) The optimizer allows the engine to take a large set of features and determine which features are more import for accurate prediction.  Which features hurt prediction.  How different features should be grouped into different sized quanta buckets.
* **Implement the optimizer / discovery aspect.**     In a Healthcare scenario the optimizer not only improves the prediction quality but it could help researchers to discover that Age is less important than they thought while Bits 1,13,27 in chromosone Y are very important for predicting Condition X when bit 1 and 13 are turned on but bit 27 is turned off.
* **Implement the Quantized Filter Algorithm in GO** This version more closely approximates the multi-layer approach used in CNN in TensorFlow but with a drastically different approach.  It can provide better output when there is a benefit from eliminating matching on a class when there is a negative match for a given feature such that there is no matching quanta from the test set.  This could allow us to do even a better job of important discovery and can provide better capacity on eliminate featurs that negatively contribute to prediction accuracy. 
* **Finish the HTTP wrappers to allow it to run as REST service.**   This could be helpful to allow the classification engine to be integrated with a VR engine running on a different piece of hardware or written in a different language. This could be important when analyzing data sets with lots of classes because the data sizes could become larger than could be contained in mobile devices. 
* **Implement Time Sequence recognition** Some gestures must start with a given pose that transitions between a series of poses ending in a termal pose. Recognizing these moving gesture recognition may be acomplished by classifying several input sets across time then running as a set through a second classifier.  Since all humans will not move at the same rate one interesting aspect will be allowing for variable time between the poses. 
* **Implement the text parsing version**.   Many common demonstration systems use text databases like IMDB and attempt to classify input text against those movies.  This is interesting but has a lot of overlap with search engines. 
* **Implement the image parsing version** so we can test against ImageNet and Minst.   Much of the TensorFlow work is demonstrated parsing and classifying images. I didn't not invent the quantized classifier for engines but it would be intersting to see how it performs. 
* **Optimize for Image Classification**   I am not sure this is worth while because image classification is one area where CNN and other Deep learning seem to perform very well with relatively little tuning. 


##Roughly prioritized Feature Work##

* Implement -class option for 3rd input file in classifyFiles

* Output should include Add recall, precision by class in classifyFiles

* Convert output of run from byte array to a structure that can be retained and used as output by optimizer to compare quality of output against other runs. 

* Classify Files needs to support both the test mode and a classify mode.

* Add Command Parser to TLearn/CNNClassify.py so it can be driven externally by command line parameters.

* Add the call to setGoEnv to all BAT that build the GO libraries.

* Modify the BAT files to skip Erase and Build when SKIPGOBUILD Env Var is set.

* Add Shell script alternatives for each of BAT file. 

* Add ability for training and classify to be ran from an array of pre-parsed float.   Need this to support speed during optimizer runs.   Ideally if input file size is below a threshold we would retain in RAM otherwise we have to scan form disk to avoid consuming all available ram. 

* Add QProb optimizer that is allowed to change Feature weight and number of buckets.

* > * Implement a option to allow the quantized classifier to store the entire array of training data in memory pre-converted to arrays of floats.  This will allow much faster re-train when changing the number number of buckets in the optimizer.    Also Requires Modify the classifier core to accept row with array of flow.  Separate the parsing / conversion form the training. 

* > * Improve optimizer specification on ClassifyFiles 

* > * Implement a -describe option to better explain reasoning output
  > * Implement option to describe high priority features  in optimizer.
  > * Implement option to describe high priority patterns for high priority features. EG: Those sets of values from the quants that deliver the greatest  predictive input for each identified class.
  > * Allow only a single feature to be retrained.  This is needed to support speed when allowing the optimizer change number of buckets for a feature. 
  > * weight and feature number of buckets seeking to maximize precision at 100% recall. Where total number of buckets is considered primary cost.  Minimum number of buckets is equal to 2.  a Feature can be turned off by setting it's feature weight to 0. 
  > * Optimizer needs ability to reserve some data from training data set to use for training.  It needs to periodically change which data is reserved.  EG: It may choose every 5th record for a while then switch to every 10th record.   It Also needs choice to use only last x% of set for optimizer setting when running with time series data.
  > *  Optimizer rules.   Can keep change if precision increases while recall stays the same.  Can keep the change is recall rises while precision remains the same. Can keep change is both precision and recall rise.   Can keep change if precision rises but recall doesn't drop below a configured threashold.    When changing number of buckets must always try  1 bucket,  1/2 current number of buckets,  random number between 1 and max buckets.   When changing  priority of a feature it must first try a priority of 0,  then a priority of 1/2 current priority, then random number between 0  and maxPriority.   When changing features the system must ensure all features are checked so first try 3 random features then 1 feature from each end working from the end towards the other end.    

* Normalize Output Probs so the sum of all classes  in any prediction for a given class is 1.0 
  But need to make sure this doesn't mess up confidence  of prediction between multiple lines.  May need to   take an approach of dividing by the number of columns  that could have contributed rather than those that  actually contributed then when we scale up it would 
  provide more accurate output.  The We are currently  apply the count for only the features that have a matching  bucket to be more accuate we need to apply feature weight
  to the divisor even if we didn't get a match for the feature  for that class.

* Finish filling in sections of the [Genomic research white paper](genomic-notes.md). Add Test set for Daily stock bar data.  Where we add a column which is a SMA(x) where X defaults to 30 days. and features are the slope of the SMA(X) when comparing current bar to SMA(x) at some number of days in past create.  In a stock scenario  you would have a goal EG: A Bar for a symbol where price rose by P% within N days without dropping by more than Q% before it reached P%.  Those that meet the rule get a class of 1 while those that fail get a class of 0.   

* Update rest of filenames links in readme.md to link to local source for the same file. 

* Add descriptions to file names in readme.md where they do not already exist or remove those files.

* Add links to bat files for new test data structures. 

* Produce GO version of the Quant Filter to see if we can improve performance on the diabetes and titanic data set.  The Quant filter is unlikely to deliver 100% recall since it aborts the match as it traverses the features when it fails to find a matching bucket id. This gives it some precision filtering capability similar to the multi-layer convoluted NN but at a lower cost.  We may be able to add probability since it is still computed by class by feaure. There some chance that more than one class will survive all layers of the filter which would mean we need to add a probability to that output to act as a tie-breaker. 
  - Add optimizer to quant filter that allows it to  vary the number of buckets by feature.  Seting number of buckets to 1 essentially turns a feature off by forcing all items into the same bucket. The primary cost function is total number of buckets. The goal of optimizer is based on starting with all buckets equal to X.  As it varies the number of buckets
    by feature it can keep the change provided it can  increase precision as long as recall does not decrease. Or it can increase recall as long as precision does not decrease recall.  We can discourage over learning by always trying a smaller number of buckets first in the optimizer.  A natural side effect of this is that the engine can turn off
    features that do not contribute either accuracy or recall.
    get. 


*   Modify Quant_prob run as server handler. 

*   Method will use main as data set name unless &dset is specified.
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

    * ​

*   Produce a version for text parsing that computes position
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

    ​

# Completed Items Phase 1 #

* DONE:2018-01-18: Update QuantProb to properly scale buckets to 
    cope with outliers to prevent them from negatively affecting 
    spread for normal distribution items.

* DONE:2017-01-20: Implement the TLearn / Tensorflow equivelant of
    classify files.  It could be named CNNClassifyFiles
    since the first implementation would use a convoluted
    neural network.

* DONE:2017-01-20 Test the classifier against Image data to see how it performs.  State of the art seems to be a 16% error rate. Will need a way to incrementally update the engine image by image rather than reading all the data from a CSV.  Try it first just by converting the images native.   Then reduce the images to greyscale and try again just to see how it does.  Will need to buy a larger hard disk.   http://image-net.org/download-imageurls  http://image-net.org/  https://en.m.wikipedia.org/wiki/ImageNet  At the very least this set of images would be great to test a image search engine.    I suspect we will need to analyze the image in smaller blocks and then classify them individually.    The best strategy would likely be to attempt to trace similar colored objects after applying an averaging filter then classify based on the vectors. http://farm1.static.flickr.com/32/50803710_8cd339faaf.jpg  As shown here differentiating the background from primary object and accomodating different scale of primary object along with different postion will be the most challenging.  

>>> I tested it with data for mnist and I cifar-10 with mnist best quantizer score for cifar was  31% compared to 36% for the Tensorflow CNN.   With mnist digit classification the best score for quantizer  was 51% while the CNN scored over 91%.    It is a little surprising the Qauntizer did as well as it did on the cifar data since it has no support for subjects moving within the frame of the data.   It could be improved but image classificaiton doesn't seem like a good place to invest with the excelent work already being done in that area.    A surprising aspect is that when n_epoch was low between 5 and 8 the CNN speed improved to be nearly comparable to the speed of the Quantized classifier.   I did find the n_epoch needed to be about 150 to get good precision out of CNN on these data sets and then it was 5 to 10 times slower. 

..

..

