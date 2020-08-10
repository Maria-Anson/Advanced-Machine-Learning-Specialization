# Question 1

## Suppose that you have a credit scoring task, where you have to create a ML model
that approximates expert evaluation of an individual's creditworthiness. Which
of the following can potentially be a data leakage? Select all that apply.

**Correct answers:**

* Data was not shuffled, this information can not be used in real-world scenario.
* Same as above, data was not shuffled, this information can not be used in
real-world scenario..

**Incorrect answers:**

* This is a perfectly fine categorical feature, don't mix it up with and ID of a
data point.

# Question 2

## What is the most foolproof way to set up a time series competition?

**Correct answers:**

* . Correct! Only complete removal of all features from test set can guarantee
that there is no data leakage.

**Incorrect answers:**

* Vulnerable to leaderboard probing.
* . Participants can try to reverse engineer time order and exploit future
peeking.

# Question 3

## Suppose that you have a binary classification task being evaluated by logloss
metric. You know that there are 10000 rows in public chunk of test set and that
constant 0.3 prediction gives the public score of 1.01. Mean of target variable
in train is 0.44. What is the mean of target variable in public part of test
data (up to 4 decimal places)?

**Correct answers:**

* Use logloss formula.

# Question 4

## Suppose that you are solving image classification task. What is the label of
this picture?

Correct answer is 3. Check image name!


