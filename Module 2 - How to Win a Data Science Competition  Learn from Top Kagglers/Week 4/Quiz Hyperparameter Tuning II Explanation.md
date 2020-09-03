# Question 1

## Which hyperparameters are first to tune in sklearn's RandomForest?

**Correct answers:**

* These parameters are important. The first one should just be sufficiently large,
you do not actually need to tune it. 

**Incorrect answers:**

* . Some of these parameters can even change the result of the training but only
because of randomness involved. They are for sure not the parameters to tune.
* . These parameters are not what you want to tune in the model!





# Question 2

## Suppose you fit LightGBM to your train data and check performance on the
validation set. The train set consists of 500 rows and 1000 different features
and validation set consist of 50 objects. You run automatic hyperparameter
optimization method overnight and in the morning you select the best parameters,
produce results for the test set and submit to the leaderboard. We also know
that test set comes from the same distribution as train and validation sets.

**Correct answers:**

* . *Correct!* This is because of [multiple comparisons
fallacy](https://en.wikipedia.org/wiki/Multiple_comparisons_problem).

**Incorrect answers:**

* The fact that test and validation sets come from the same distribution does not
make them similar in terms of optimal hyperparameters.





# Question 3

## Suppose you want to find a good set of hyperparameters for a dataset with 1000
points and have resources to do fitting 2000 times. Which method of model
selection your should use?

**Correct answers:**

* Yes! It is easy to note than we will check 2000 / k sets of hyperparameters --
this number still big enough (for typical k in [3,..,10]), but this scheme is
much more robust.

**Incorrect answers:**

* See previous question for detailed explanation.
* This is overfitting. Such way cannot be used to estimating quality of model.
* Since LOO requires 1000 fittings for single set of hyperparameters, we can
estimate only 2 combinations of hyperparameters -- it's way too small.





# Question 4

## Suppose you train Neural Network with SGD and see that it overfits data. Which
of the following actions can help you to regularize model?

**Correct answers:**

* Yes, Dropout is a known way to regularize model via dropping some randomly
selected features at every step.
* Weight Decay is essentially the same thing as L2 Regularization.
* Reducing number of parameters make model less flexible and harder to overfit

**Incorrect answers:**

* .  Adam is a better optimizer than SGD, thus overfits training data better and
faster. 
* No, this will increase complexity of neural net and do not introduce any
regularization effect.






