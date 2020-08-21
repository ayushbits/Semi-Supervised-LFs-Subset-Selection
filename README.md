# Requirements
This code has been developed with
  - python 3.6
  - tensorflow 1.14.0
  - numpy 1.17.4
  - torch 1.1.0

# Data Description
The Data/ directory contains dataset for the following 3 datasets:
  * MITR - Slot filling task (Source: https://groups.csail.mit.edu/sls/downloads/restaurant/)
  * YOUTUBE - Spam Classification task of youtube comments (Source: http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection)
  * SMS - Spam classification task of text messages (Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
  * CENSUS 
  
## data/SMS (or any other data dir) consists following four pickle files
  * d_processed.p (d set: labeled data ) 
  * U_processed.p (U set: unlabeled data)
  * test_processed.p (test data)
  * **NOTE** U_processed.p for YOUTUBE and MITR is unavailable on GitHub due to larger size. You can download entire data dir from [this](https://drive.google.com/file/d/1dWTUC_f0Ks-Hg4TtkHrwWy0RmzLMH-X1/view?usp=sharing) link

## Following objects are dumped inside each pickle file
* x : feature representation of instances
    - shape : [num_instances, num_features]
* l : Class Labels assigned by rules
    - shape : [num_instances, num_rules]
    - class labels belong to {0, 1, 2, .. num_classes-1}
    - l[i][j] provides the class label provided by jth rule on ith instance
    - if jth rule doesn't cover ith instance, then l[i][j] = num_classes (convention)
    - in snorkel, convention is to keep l[i][j] = -1, if jth rule doesn't cover ith instance
* m : Rule coverage mask
    - A binary matrix of shape [num_instances, num_rules]
    - m[i][j] = 1 if jth rule cover ith instance
    - m[i][j] = 0 otherwise
* L : Instance labels
    - shape : [num_instances, 1]
    - L[i] = label of ith instance, if label is available i.e. if instance is from labeled set d
    - Else, L[i] = num_clases if instances comes from the unlabeled set U
    - class labels belong to {0, 1, 2, .. num_classes-1}
* d : binary matrix of shape [num_instances, 1]
    - d[i]=1 if instance belongs to labeled data (d), d[i]=0 otherwise
    - d[i]=1 for all instances is from d_processed.p
    - d[i]=0 for all instances in other 3 pickles {U,validation,test}_processed.p
* r : A binary matrix of shape [num_instances, num_rules]
    - r[i][j]=1 if jth rule was associated with ith instance
    - Highly sparse matrix
    - r is a 0 matrix in all the pickles except d_processed.p
    - Note that this is different from rule coverage mask "m"
    - This matrix defines the coupled rule,example pairs.

# Usage
  - Run respective .sh files to train the model
  - To run semi-supervised model of youtube - run *tr_youtube.sh*
