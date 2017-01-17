#Basics of a Quantized Machine Learning Classifier#

#Binome Prediction when using output from several tools#

#Genome Prediction of Outcomes #
  such as Risk for Breast Cancer#
  I am not a Genome expert but I have interacted with people who 
  work on Genomic tools and have picked a up a little knowledge
  along the way.  Please correct my vocabulator and explanation
  to be accurate.
  
  As I understand the genome if it is stretched out it can be 
  modeled as a series of of 0 and 1 stretched out in a very long 
  array.  
   
  Researchers seem to identify suspect regions of this data 
  they then analalyze for patterns that they use to predict
  against a class such as Has Breast Cancer = 1,  
  Does not have Breast Cancer = 0.
  Or they attempt to predict High cancer risk, Low cancer risk.   
  
  
>### What is a Feature ###
>> In the context of a genome each unique 0 or 1 in 
>> the squence is considered a facet.   If you think of
>> it as a CSV file the the first column would be used
>> as a predictor then there would be 1..N columns 
>> each representing a unit bit of data.
>>      
>> More complex models may assign a number between 0 and
>> X for each position in the genome.  
>>     
>> A slice of Genomic data can be thought of as Comma 
>> delimited list like:
>>> * 0,0,1,0,1,1,1,1,1,0,0,1,1,0,1
>>      
>> The slice could be pulled from a single portion of
>> the genome or could be pulled from several different
>> sections and assembled as one longer slice.   The first
>> number is generally the measured outcome or class while
>> others are measures from the data.  In some instances
>> data can be added from other sources such as: 
>>     
>> * 0,40,2,60,0,1,0,1,1,1,1,1,0,0,1,1,0,1
>>        
>>   * Colmmn 0 = Measured Outcome or Class
>>   * Column 1 = Age, 
>>   * Column 2 = Number of exercise days per week,
>>   * Column 3 = Resting heart beat. 
>>   * Rest are genome measures.         
      
### What is a Class ###
      Each of the unique outcomes they are measuring
      represent one class.   For example:
        * class = 1 = Person had heart attack before Age of 40
        * class = 0 = Person did not have heart attack before 40.
      A more complex set outcomes
        * class = 0 = Diagnosed Parkinsons Age 0 .. 20
        * class = 1 = Diagnosed Parkinsons Age 21 .. 40
        * Class = 2 = Diagnosed Parkinsons Age 41 .. 50
        * class = 3 = Diagnosed Parkinsons Age 51 .. 60
        * class = 4 = Diagnosed Parkinsons Age 61 .. 90        
  
      In reality there can be any number up classes but
      keeping the number small tends to work somewhat better
      but in some applications the classes could be ethnic 
      origin which may require hundreds of classes. 
  
  While input data like a Genomome measures individual 
  data elements as a zero or one.  We can also use other
  data such as Age that may be a number between 0.0 and
  150.  In low dimensionality data such as zero or 1 
  the problem is somewhat easier but in age we may find
  that dividing the data into groups can quanta buckets
  such as 0..25 = young,  26..50=adult,  51..65=mature,
  65..80=senior,  81-150=ancient. Grouping the data 
  for similarity can allow better predictive output 
  with less input data.  Choosing the number of buckets
  is as much art as science and is covered in another topic
  in some instances the system is allowed to choose the
  number of buckets to find the best predictive input
  which is normally done by an optimizer.  In generall
  reducing the number of measurement risks reduce risk
  of over learning which is covered in another topic. 
  
  ## Chosing Input Data ##
    Some of genome analysis tools are written with specific
    algorithms based on domain expertise.  Machine learning
    attempts to accomplish the same thing using pattern 
    matching and statistics but it still helpful to have human
    experts who can identify the outcomes they want to measures
    and isolate portions of the genome.   Each different person
    becomes one row of data.   In some instances we directly
    consume the genomic data slice while in others we consume
    data produced by several other previous analysis tools.  
    The practitioner may also choose to use some genome data
    for one or more slice plus the output from one or more
    previous analysis tools.   

  ## What are Classes ##
    When using machine learning we take a set of rows where 
    the outcome is known.  Each row represents the data from
    one genome or one genomic run.   Sample outcomes could be
    High Cancer Risk=0, Low cancer Risk = 1.    We call each 
    of these unique outcomes a class.  There must be at least 
    two classes but there could be a larger numer.  The general 
    rule is more classes will require more training data. 
    Choosing what they want to measure or predict is part
    of the art of data science. One of the hardest things in 
    machine learning is gathering a sufficient amount of 
    data that has been pre-classified to use for training. 
  
  ## Test Versus Learn Verus Runtime ##
    The Data set is divided into two sets.  One portion is 
    used for training the system.  The second set is used to 
    test the systems ability to accurately classify outcomes
    given a set of inputs.   We measure two main aspects 
    precision which is if the system classifies a given row
    is a member of a given class such as 1 = high risk for 
    X how often is it correct.    The second
    Recall is of those who should have been classified as
    1 = high rist for X how many did it find.    There is 
    always a balancing act where increasing precision will
    reduce recall or visa-versus.  
    
    Optimizers are used in an attempt to increase either 
    precision or recall while minimizing detrimental 
    effect on the other.  A simple optimizer may vary
    the weight or important of individual features 
    In some instances the choices 
    of the optimizer 
  
  #Some Data is more important than others# 
  
  We take whatever slice of the genome the researcher thinks
  might may have important data along with the diagnosed 
  class.  Once the system is trained then the system can 
  be tested against new genome input using the same slice
  of data and will attempt to predict the class of the 
  output.   We Take the majority of data and use it for 
  training and then 

  For machine learning we would need at least 1K rows (50K more ideal) one row per person.  The data would be tagged as - Has X yes=1, no=1.   The engine should be able to learn from this and predict yes,no if you feed it the same sub region of the genome for each person.   If you could provide a set of data like this I would love to test the idea. 
  
  # Genone Use of Optimizers #
  
  # Prepare Data #
  

#Genome Which chromosones are most important predictors#

#Minimizing Overlearning in Optimizer#

#Choosing Number of Quanta Buckets.  More is not always better#


#Outliers Detection & Handling#
One possible problem with a quantized engine is that if the 
data set is noisy and contains data for any feature that is 
abnormally high or low this can create a large range which can
cause the quanta buckets to be created larger than desirable.

An example of this is that stock values move day to day a 
relatively small amount normally under 1%. Every so often
a price could move by a larger amount such as 12%.   When 
used for prediction we are more interested in the movements 
of 97% of all bars so we want the numeric range used to compute 
the quanta size to reflect the min, max values of those 97%
of bars closer to the center of the distribution of values.
If the average daily movement is 1% and we are using a 10 
quanta system the bucket size would be 0.1%.  If we do not
suppress the influence of outliers we would use a 12% range
so each quanta would be sized at 1.2% which would make all 
the normal data values clump together into a single quanta 
bucket which would remove the usefulness as predictive values. 

To remove undue influence from the outliers we want must 
detect the values of the top and bottom extreme values which 
normally requires sorting all the training values.  Since our
training data set will likely exceed available ram the load 
and sort does not work.  

What we do is take the absolute
minimum and absolute maximum values to divide each value read
for a each feature evently between 1,000 buckets.  We keep 
a separate set of counts called a distribution matrix for 
each feature.  

We can scan from the Bottom of the distribution matrix
accumulating counts until we have enough records to represent 1.5% 
of the data and then scan from the max value down. 

Once we find the distribution buckets with the outlier values
eliminatedit becomes a matter of simple math
to compute new effective minimum and effective maximum values.
We use the new effictie min/max values to compute new quana 
bucket sizes.  The actual buckets are based on a sparse matrix 
so the extreme values get large bucket Id
but that is actually desired.

The system defaults to setting the effective range by removing the
influence from 1.5% of the total row set from each end.  This can be 
change if needed.  The actual code is implemented in 
[src/qprob/classify.go](src/qprob/classify.go) in the function 
setFeatEffMinMax but it is enabled by the 1000 bucket distribution
grid that is built in [/src/qprob/csvInfo.go](/src/qprob/csvInfo.go)
in the method BuildDistMatrix().


#Temporal Reinforcement#

  One of the features provided by convoluted neural networks is the 
  ability to have some more recent records influence classification 
  results more than other records.   
  
  This can be supported for 
  Quantized probability system by allowing the system train against
  the entire set and then re-train with more recent data multiple times.
  
  The system computes a probability value for each feature based on how
  many times the bucket value occurred for a given class divided by the 
  total number of records.  By applying more recent data many times 
  it increases both the class counts favoured by the most recent data 
  plus the total record count.  This will give those newer records a higher 
  probability input.   

   * An extension to this could be to look at the span of
     of time covered in the training set up front and automatically adjust
     the count using a multiplier as we move through the training data
     from old to new.  This would be a relatively easy
     enhancement to add. 

     
     