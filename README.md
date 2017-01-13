# Simplified Static Gesture classifier  #

See Overview.pdf in this repository for conceptual overview of the approach.



### What is this repository for? ###

* Quick summary
* 0.1

* License (MIT) We do sell consulting services http://BayesAnalytic.com/contact

### Basic Contents ###
 *data/train/gest_train_ratio2.csv - Input training data used for these tests.  We need thousands of     additional training samples feel free to volunteer after your read overview.pdf in this repository.
 * quant_filt.py  - Machine learning Quantized filter classifier.  This system can provide very 
   fast classificaiton with moderate memory use and it is easy to see how likely the match is to 
   be accurate.

   * quant_prob.py - Machine learning Quantized probability classifier. Not quite as precise under
   some conditions and quant_filt.py but it can cope with greater amounts of training noise while
   still delivering good results with moderate amounts of training data.  
   
 * tlearn/simple_gestures.py - sample of reading CSV to to train TensorFlow Model. 
   Unfortunately this program while it runs does a pour job of classification. I think 
   this is the result of insufficient training data but there is a chance that I still have
   a bug in the interface ot TensorFlow. 

 * tlearn/tensFlowReadme.docx - Notes I made while getting tensor flow running on my windows laptop.

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