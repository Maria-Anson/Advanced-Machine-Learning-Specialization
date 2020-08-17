## Metrics
> 
> Total points 6
> 
> 1.
> 
> Question 1
> 
> Suppose we solve a binary classification task and our solution is scores with logloss. What predictions are more preferable in terms of logloss if true labels are y_true = [0, 0, 0, 0].
> 
> 1 point
> 

      y_pred = [0.5, 0.5, 0.5, 0.5] 
> 
>  y_pred = [0, 0, 0, 1] 
> 
>  y_pred = [0.4, 0.5, 0.5, 0.6] 
> 
> 2.
> 
> Question 2
> 
> Suppose we solve a regression task and we optimize MSE error. If we managed to lower down MSE loss on either train set or test set, how did we change [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) coefficient between target vector and the predictions on the same set?
> 
> 1 point
> 
>  The correlation did not change. 
> 
>  The correlation became larger. 
> 
>  The correlation was also lowered. 
> 

      Any behavior is possible. 
> 
> 3.
> 
> Question 3
> 
> What would be a best constant prediction for a following multi-class classification task with 4 classes? The solution is scored with multi-class logloss. The number of objects of each class in train set is: 18, 3, 15, 24\.
> 
> Enter four comma separated values. Round each to two decimal places and use a leading zero before a fractional part (e.g. "0.50"; not ".5").
> 
> 1 point
> 
> Enter answer here

      0.3,0.05,0.25,0.4
> 
> 4.
> 
> Question 4
> 
> What is the best constant predictor for R-squared metric?
> 
> 1 point
> 
>  One minus target mean 
> 
>  0.5 
> 
>  (Log of target mean) + 1 
> 

      Target mean 
> 
>  Target mean divided by target variance 
> 
> 5.
> 
> Question 5
> 
> Select the correct statements.
> 
> 1 point
> 
>  Optimization loss is always different to target metric. 
> 

      Optimization loss can different to target metric. 
> 

      Optimization loss can be the same as target metric. 
> 
>  Optimization loss is always the same as target metric. 
> 
> 6.
> 
> Question 6
> 
> Suppose the target metric is **M1**, and optimization loss is **M2**. We train a model and monitor its quality on a _holdout set_ using metrics **M1** and **M2**.
> 
> Select the correct statements.
> 
> 1 point
> 

      There is no definite relation between the best iterations for **M1** score and **M2** score. 
> 
>  If the best **M1** score is attained at iteration N, then the best **M2** score is always attained after N-th iteration. 
> 
>  If the best **M1** score is attained at iteration N, then the best **M2** score is always attained before N-th iteration. 
> 
>  If the best **M1** score is attained at iteration N, then the best **M2** score is always attained also at the iteration N.
>
> -- https://www.coursera.org/learn/competitive-data-science/exam/x1VBU/metrics/attempt#Tunnel Vision Close
