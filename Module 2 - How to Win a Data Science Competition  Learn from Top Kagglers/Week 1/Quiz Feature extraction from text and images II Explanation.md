# Question 1

## Select true statements about n-grams.

**Correct answers:**

* . Correct, because ngrams encode sequences of words.
* . Correct. Ngrams deal with counts of words occurrences, and not every word can
be found in a document. For example, if we count occurrences of words from an
english dictionary in our everyday speech, a lot of words won't be there, and
that is sparsity.

**Incorrect answers:**

* . No, ngrams deals with words occurrences and not their importance.
* . Although, there is Levenshtein distance, there is no such thing as
Levenshteining.

# 



# Question 2

## Select true statements.

**Correct answers:**

* . Correct! Number of features in Bag of words approach is usually equal to
number of unique words, while number of features in w2v is restricted to a
constant, like 300 or so.
* Correct. This is one of the main benefits of w2v in competitions.

**Incorrect answers:**

* . Incorrect. Meaning of a value in BOW matrix is the number of a word's
occurrences in a document.
* Incorrect. Both approaches are valuable and you should try to utilize both of
them.

# 



# Question 3

## Suppose in a new competition we are given a dataset of 2D medical images. We
want to extract image descriptors from a hidden layer of a neural network
pretrained on the ImageNet dataset. We will then use extracted descriptors to
train a simple logistic regression model to classify images from our dataset. 

## We consider to use two networks: ResNet-50 with imagenet accuracy of X and
VGG-16 with imageNet accuracy of Y (X < Y). Select true statements.

**Correct answers:**

* We should evaluate both. Correct! This depends on the a specific dataset and a
specific task, so you should evaluate both!

**Incorrect answers:**

* . Incorrect. With one CNN you can get different descriptors from different
layers.
* . Incorrect. Although, ResNet50 shows better performance on Imagenet, this
depends on the a specific dataset and a specific task.
* . Incorrect in general. Moreover it is hard to come up with an image that will
have the same descriptors in both networks.
* . Incorrect. This depends on the a specific dataset and a specific task.



# 

# Question 4

## Data augmentation can be used at (1) train time (2) test time

**Correct answer:** 

. Data augmentation can be used (1) to increase the amount of training data and
(2) to average predictions for one augmented sample.


