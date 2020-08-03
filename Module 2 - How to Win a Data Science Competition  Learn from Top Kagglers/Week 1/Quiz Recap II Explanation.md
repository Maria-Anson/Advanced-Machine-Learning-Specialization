# Question 1

## What back-propagation is usually used for in neural networks?

**Correct answer: **

* 

**Incorrect answers:**

* This is called "forward pass"
* This one doesn't involve gradients and have nothing to do with back-propagation
* In back-propagation gradients are calculated exactly, not random

# 



# Question 2

## Suppose we've trained a RandomForest model with 100 trees. Consider two cases:

## 1. We drop the first tree in the model

## 2. We drop the last tree in the model

## We then compare models performance *on the train set*. Select the right answer.

**Correct**** answers:**

* In RandomForest model we average 100 similar performing trees, trained
independently. So the order of trees does not matter in RandomForest and
performance drop will be very similar on average.

**Incorrect answers:**

* In RandomForest model we average 100 similar performing trees, trained
independently. So the order of trees does not matter in RandomForest.
* . Similar to the previous one.





# Question 3

## Suppose we've trained a GBDT model with 100 trees with a fairly high learning
rate. Consider two cases:

## 1. We drop the first tree in the model

## 2. We drop the last tree in the model

## We then compare models performance *on the train set*. Select the right answer.

**Correct**** answers:**

* In GBDT model we have sequence of trees, each improve predictions of all
previous. So, if we drop first tree — sum of all the rest trees will be biased
and overall performance should drop. If we drop the last tree -- sum of all
previous tree won't be affected, so performance will change insignificantly (in
case we have enough trees)

**Incorrect answers:**

* 
* 





# Question 4

## Consider the two cases:

## 1. We fit two RandomForestClassifiers 500 trees each and average their predicted
probabilities on the test set.

## 2. We fit a RandomForestClassifier with 1000 trees and use it to get test set
probabilities.

## All hyperparameters except number of trees are the same for all models.Select
the right answer.

**Correct answers:**

* Each tree in forest is independent from the others, so two RF with 500 trees is
essentially the same as single RF model with 1000 trees

**Incorrect answers:**

* 
* 





# Question 5

## What model was most probably used to produce such decision surface? Color (from
white to purple) shows predicted probability for a point to be of class "red".

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_LUm1XqXEeeA3RJRlG3Uqg_56a252e3d83d1655792bfd4da667310e_dt.png?expiry=1596585600000&hmac=HCBa0ycYfssu-11ZzYXHlExSmPGPzQq-TTU0PMyEU30)

**Correct**** answers:**

* Decision surface consists of lines parallel to the axis and it is sharp.

**Incorrect answers:**

* Decision surface is not linear.
* Decision surface consists of lines parallel to the axis and it is sharp -- in
case of RF boundaries should be much more shooth.
* Decision surface doesn't depend on distance from objects

# 



# Question 6

## What model was most probably used to produce such decision surface?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ntCFVnqZEee6QxIw_sWf5g_1d788ff8639da8a2e6c564ae12e6f13a_rf.png?expiry=1596585600000&hmac=c4i8kEIJAtfEZlMthIGR5hMtt5ZfAB0lUUq4-vl3UNE)

**Correct**** answers:**

* Decision surface consists of lines parallel to the axis and its boundaries are
smooth

**Incorrect answers:**

* Decision surface is not linear
* Decision surface consists of lines parallel to the axis and it is *not* sharp
* Decision surface doesn't depend on distance from objects




