#TODO - Roughly prioritized Features#
Please coment and critique.  I need your help to ensure
future work is prioritized to help the people actually 
using the library.  http://BayesAnalytic.com/contact



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
  
* Update rest of filenames links in readme.md to link to
  local source for the same file. 
  
* Add descriptions to file names in readme.md where they do 
  not already exist or remove those files.
  
* Add links to bat files for new test data structures. 
  
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

