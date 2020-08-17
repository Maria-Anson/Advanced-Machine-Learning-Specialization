# Question 1

## What can be an indicator of usefulness of mean encodings?

**Correct answers:**

* 

**Incorrect answers:**

* This is not an indicator because the majority of ML models deal with binary
variables just fine. But keep in mind that there could be special cases when
encoding binary variables with target mean may be useful. For example, KNN
models might work better on mean-encoded binary features.
* There is no connection between mean encodings and learning to rank tasks.

# Question 2

## What is the purpose of regularization in case of mean encodings?

**Correct answers:**

* 
* Only with regularization we can use mean encodings to the fullest.

**Incorrect answers:**

* Don't mix it up with L1 penalty.

# Question 3

## What is the correct way of validation when doing mean encodings?

**Correct answers:**

* That way we avoid target variable leakage.

**Incorrect answers:**

* This way we will overfit, because target from validation is used to calculate
mean encodings on train. So the model implicitly uses this information which
results in mild target leakage.
* This way we will overfit even more, because target from validation is explicitly
used to calculate mean encodings.

# Question 4

## Suppose we have a data frame 'df' with categorical variable 'item_id' and target
variable 'target'.We create 2 different mean encodings:

## 1)via df['item_id_encoded1'] = df.groupby('item_id')['target'].transform('mean')

## 2)via OneHotEncoding item_id, fitting Linear Regression on one hot-encoded
version of item_id and then calculating 'item_id_encoded2' as a prediction from
this linear regression on the same data.

**Correct answers:**

* Remember, after one hot encoding there will be only one '1' in each row and the
rest - zeros. Since we don't have a regularization, coefficients of each
variable would be target means of item_id corresponding to that variable. 

**Incorrect answers:**

* Thats not always true. With regularization it will not converge to the same
values as 'item_id_encoded1'
* No, it has nothing to do with categorical sizes.




