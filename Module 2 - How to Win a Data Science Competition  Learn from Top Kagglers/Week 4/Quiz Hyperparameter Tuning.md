### Practice quiz
> 
> Total points 3
> 
> 1.
> 
> Question 1
> 
> What can we do to deal with overfitting in LightGBM?
> 
> 1 / 1 point
> 
>  Decrease `min_data_in_leaf` 
> 
>  Increase `bagging_fraction` 
> 
>  Increase `bagging_seed` 
> 

      Decrease `num_leaves` 
> 
> Check
> 
> Correct
> 
> In fact, higher `num_leaves` more complex trees GBM can build.
> 

      Increase `min_data_in_leaf` 
> 
> Check
> 
> Correct
> 
> Yes, this parameter serves as regularization parameter: the higher `min_data_in_leaf`, more regularized the model.
> 
> 2.
> 
> Question 2
> 
> What of the following parameter changes speed-up _training process_ for LightGBM? That is, time needed for one boosting iteration is decreased.
> 
> 1 / 1 point
> 
>  Increasing max_bin 
> 
>  Decreasing verbosity 
> 

      Decreasing bagging_fraction 
> 
> Check
> 
> Correct
> 
> Yes! This parameter controls fraction of objects, that will be used at every boosting iteration.
> 

      Decreasing num_leaves 
> 
> Check
> 
> Correct
> 
> Correct! The number of splits in a tree is very dependent on num_leaves parameter.
> 
> 3.
> 
> Question 3
> 
> How usually training time of SVM changes if we increase the value of parameter "C"?
> 
> 1 / 1 point
> 
>  Training time decreases 
> 

      Training time increases 
> 
> Check
> 
> Correct
> 
> Yes! That is why it is better to start with small values of "C"
>
> -- https://www.coursera.org/learn/competitive-data-science/quiz/k7xTP/practice-quiz/attempt?redirectToCover=true#Tunnel Vision Close
