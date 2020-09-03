## Graded quiz
> 
> Total points 4
> 
> 1.
> 
> Question 1
> 
> Which hyperparameters are first to tune in sklearn's RandomForest?
> 
> 1 point
> 

      n_estimators, max_depth, min_samples_split. 
> 
>  n_jobs, random_state, verbose. 
> 
>  bootstrap, oob_score, warm_start. 
> 
> 2.
> 
> Question 2
> 
> Suppose you fit LightGBM to your train data and check performance on the validation set. The train set consists of 500 rows and 1000 different features and validation set consist of 50 objects. You run automatic hyperparameter optimization method overnight and in the morning you select the best parameters, produce results for the test set and submit to the leaderboard. We also know that test set comes from the same distribution as train and validation sets.
> 
> 1 point
> 
>  There is a low chance of overfitting to the validation set. That is, there is a high chance that score on the test set will be good. This is because we found good parameters on validation set and test set is similar to the test set. 
> 

      There is a high chance of overfitting to the validation set. That is, there is a high chance that score on the test set will be bad. This is because we've tried too much hyperparameters while the dataset is small and the number of features is large. 
> 
> 3.
> 
> Question 3
> 
> Suppose you want to find a good set of hyperparameters for a dataset with 1000 points and have resources to do fitting 2000 times. Which method of model selection your should use?
> 
> 1 point
> 
>  Select model by quality on training set (i.e. fit model on whole dataset and measure quality on the same data). 
> 
>  Hold-Out scheme (i.e. divide data into two parts, use first for model fitting and second for estimation of quality). 
> 
>  Leave-one-out validation (i.e. fit model for all points except one and estimate quality for this single point; repeat for every point). 
> 

      k-Fold scheme (i.e. split data into k part, use k-1 parts for training and the last one for quality estimation; repeat for each part) 
> 
> 4.
> 
> Question 4
> 
> Suppose you train Neural Network with SGD and see that it overfits data. Which of the following actions can help you to regularize model?
> 
> 1 point
> 
>  Change optimization method to Adam. 
> 
>  Add more layers. 
> 

      Reduce number of parameters (e.g. remove some layers) 
> 

      Add (or increase) Weight Decay. 
> 

      Insert (or increase rate of) Dropout layers inside NN.
>
> -- https://www.coursera.org/learn/competitive-data-science/exam/ZzxU7/graded-quiz/attempt#Tunnel Vision Close
