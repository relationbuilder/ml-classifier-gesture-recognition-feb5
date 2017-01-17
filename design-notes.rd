

##Temporal Reinforcement##

* One of the featurs provided by convoluted neural networks is the 
  ability to have some more recent records influence classification 
  results more than other records.   
  
  This can be supported for 
  Quantized probability system by allowing the system train against
  the entire set and then re-train with more recent data multiple times.
  The system computes a probability value for each feature based on how
  many times the bucket value occured for a given class divided by the 
  total number of records.  By applying more recent data many times 
  it increases both the class counts favoured by the most recent data 
  plus the total record count.  This will give those newer records a higher 
  probability input.   

   * A extension to this could be to look at the span of
     of time covered in the training set up front and automatically adjust this
     multiplier as you move from old to new data.  This would be a relatively easy
     enhancement to add. 